import asyncio, ast, base64, io, ray, re, math
from typing import Dict, Any, List, Tuple, Optional
import os, pathlib, json
from PIL import Image
import time, datetime
import logging
from functools import partial
from io import BytesIO

from desktop_env.desktop_env import DesktopEnv
from src.utils.ui_tars_utils import *
from src.utils.log_config import setup_logging
from src.utils.kill_all_env import stop_emulator
from src.core.scoring_switch import resolve_process_reward_scoring
from src.core.seekjudge_rl import evaluate_seekjudge_process_reward
from copy import deepcopy
from dotenv import load_dotenv
import json
from openai import OpenAI

class ResourceExhaustedException(Exception):
    """Resource exhaustion exception"""
    pass


def _emulator_id_for_log(env) -> Optional[str]:
    provider = getattr(env, "provider", None) if env is not None else None
    return getattr(provider, "emulator_id", None)

# Setup unified logging system
setup_logging()
logger = logging.getLogger(__name__)


def _raw_process_reward_payload(step_scores, group_scores):
    """Keep PR components separate in SQL while retaining step alignment."""
    return {
        "format": "raw_pr_v1",
        "step_scores": step_scores,
        "group_scores": group_scores,
    }


def _valid_process_reward(process_reward):
    """Validate both legacy lists and raw component payloads before insertion."""
    if isinstance(process_reward, dict):
        step_scores = process_reward.get("step_scores")
        group_scores = process_reward.get("group_scores")
        if not isinstance(step_scores, list) or not isinstance(group_scores, list):
            return False
        if len(step_scores) == 0 or len(step_scores) != len(group_scores):
            return False
        values = step_scores + group_scores
    elif isinstance(process_reward, list):
        if not process_reward:
            return False
        values = process_reward
    else:
        return False
    try:
        return all(float(value) >= 0 for value in values)
    except (TypeError, ValueError):
        return False

"""
Qwen3VL-compatible trajectory runner.

This keeps the rollout lifecycle from trajectory_runner.py, but switches the
model-facing protocol to the OSWorld Qwen3VL function-call prompt.
"""

def pil_to_base64(img_bytes: bytes) -> str:
    return base64.b64encode(img_bytes).decode("utf-8")


def py_string(text: str) -> str:
    return json.dumps("" if text is None else str(text), ensure_ascii=False)

@ray.remote
class TrajectoryRunnerActor:
    def __init__(self, task_info: Dict[str, Any], runner_cfg, process_reward_cfg=None):
        
        # --- load config ---
        self.runner_cfg = runner_cfg
        self.max_images = self.runner_cfg.max_images
        self.max_tests = self.runner_cfg.max_texts
        self.max_steps = self.runner_cfg.max_steps
        self.root = self.runner_cfg.root
        self.process_reward_cfg = process_reward_cfg
        self.pr_backend = str(getattr(process_reward_cfg, "backend", "disabled")).lower()
        if self.pr_backend not in {"disabled", "original", "seekjudge"}:
            raise ValueError(f"Unsupported process_reward.backend: {self.pr_backend}")
        self.pr_mode = str(getattr(process_reward_cfg, "mode", "single")).lower()
        self.enable_pr_phase3 = bool(getattr(process_reward_cfg, "enable_phase3", False))
        self.pr_scoring = resolve_process_reward_scoring(
            getattr(process_reward_cfg, "scoring_version", "v2_1")
        )
        # self.save_img_pt = self.runner_cfg.save_img_pt
        # self.rollout_n = self.runner_cfg.rollout_n

        # --- process task info for running ---
        self.task_cfg: Dict[str, Any] = task_info["task_config"]
        self.task_id = self.task_cfg["raw"]["task_id"] 
        self.trace_id = task_info.get("trace_id", "unknown_trace")
        self.instruction = task_info.get("instruction") or self.task_cfg.get("instruction", "")
        self.coordinate_grid = int(getattr(self.runner_cfg, "qwen3_coordinate_grid", 1000))
        self.qwen3_history_n = int(getattr(self.runner_cfg, "qwen3_history_n", 100))
        self.qwen3_image_max = int(getattr(self.runner_cfg, "qwen3_image_max", 10))
        self.qwen3_fold_size = int(getattr(self.runner_cfg, "qwen3_fold_size", 1))
        self.qwen3_collapse_text = str(getattr(self.runner_cfg, "qwen3_collapse_text", "This screenshot has been collapsed."))
        self.qwen3_enable_thinking = bool(getattr(self.runner_cfg, "qwen3_enable_thinking", False))
        self.folded_prefix_k = 0
        self.processed_screen_size: Tuple[int, int] = (1000, 1000)

        clean_instruction = self.instruction.split("Here is an instruction to help you complete the task: \n", 1)[0]
        self.clean_instruction = clean_instruction
        self.base_messages: List = self._build_qwen3_base_messages(clean_instruction)
        self.base_messages_for_save: List = deepcopy(self.base_messages)
        
        self.task_root = f"{self.task_id}_{self.trace_id}"

        self.prompt_dialogue: List[Dict] = []   # for model client
        self.save_dialogue:   List[Dict] = []   # for partial storage
        self.save_dialogue_full: List[Dict] = []    # for full storage
        self.image_refs: List[Dict] = []  # record image position
        self.screenshots: List[str] = []
        self.screenshot_paths: List[str] = []
        self.responses: List[str] = []
        self.actions: List[str] = []
        self.parsed_actions: List[Dict[str, Any]] = []

        # --- initialize env ---
        self.env = None    # will be defined in _init_env()     
        self.env_ready = asyncio.Event()
        self.env_init_success = False  # Indicates whether environment initialization was successful
        self.env_init_timeout = 1200.0  # Environment initialization timeout
        self.env_init_start_time = None
        self.env_timeout_task = None  # Environment timeout monitoring task
        # self.env_init_end_time = None
        
    
    def get_service_id(self) -> Optional[str]:
        """Get environment service_id"""
        if not self.env_init_success:
            logger.warning(f"[{self.trace_id}] Environment initialization failed, cannot get service_id - task_id: {self.task_id}")
            return None
        elif self.env is None or getattr(self.env, "provider", None) is None:
            logger.warning(f"[{self.trace_id}] Environment not initialized, cannot get service_id - task_id: {self.task_id}")
            return None
        elif self.env.provider.emulator_id is None:
            logger.warning(f"[{self.trace_id}] Environment emulator_id is empty - task_id: {self.task_id}")
            return None
        else:
            logger.info(f"[{self.trace_id}] Successfully obtained service_id: {self.env.provider.emulator_id} - task_id: {self.task_id}")
            return self.env.provider.emulator_id

    
    
    async def wait_for_env_ready(self, timeout: float = 120.0) -> bool:
        """Wait for environment initialization to complete (for external calls)"""
        try:
            await asyncio.wait_for(self.env_ready.wait(), timeout=timeout)
            return self.env_init_success
        except asyncio.TimeoutError:
            return False

    def is_env_ready(self) -> bool:
        """Check if environment initialization was successful"""
        return self.env_init_success

    async def monitor_env_timeout(self):
        """Monitor environment initialization timeout, automatically close environment if timeout"""
        if self.env_init_start_time is None:
            logger.warning(f"[{self.trace_id}] env_init_start_time not set, cannot monitor timeout - task_id: {self.task_id}")
            return
        
        check_interval = 60.0  # Check every minute
        while True:
            await asyncio.sleep(check_interval)
            
            if self.env is None:
                # Environment has been closed, exit monitoring
                break
            
            elapsed_time = time.time() - self.env_init_start_time
            if elapsed_time > self.env_init_timeout:
                logger.warning(f"[{self.trace_id}] Environment runtime timeout - task_id: {self.task_id}, "
                            f"Elapsed time: {elapsed_time:.2f}s, Timeout threshold: {self.env_init_timeout}s")
                try:
                    if self.env is not None:
                        self.env.close()
                        logger.info(f"[{self.trace_id}] Environment closed due to timeout - task_id: {self.task_id}")
                except Exception as e:
                    logger.error(f"[{self.trace_id}] Error closing environment due to timeout - task_id: {self.task_id}, Error: {e}")
                finally:
                    self.env = None
                break

    async def run_episode(self, model_pool, storage, mysql_writer, pr_model_pool=None):
        """
        Single task rollout loop
        """
        # --- save task config ---
        storage_root = await storage.save_task_config.remote(self.task_root, self.task_cfg)

        logger.info(f"[{self.trace_id}] TrajectoryRunnerActor initialized - task_id: {self.task_id}, task_root: {self.task_root}")
        
        try:
            # --- initialize env and first frame ----
            try:
                # self._init_env(self.runner_cfg.env)
                await asyncio.to_thread(self._init_env, self.runner_cfg.env)
                self.env_init_success = True
                self.env_init_start_time = time.time()
                # Start environment timeout monitoring task
                # self.env_timeout_task = asyncio.create_task(self.monitor_env_timeout())
                logger.info(f"[{self.trace_id}] Environment initialization successful - task_id: {self.task_id}")
            except Exception as e:
                self.env_init_success = False
                init_error = e
                logger.error(f"[{self.trace_id}] Environment initialization failed - task_id: {self.task_id}, Error: {init_error}")
                try:
                    task_error_path = os.path.join(storage_root, f"task_{self.task_id}_trace_{self.trace_id}.json")
                    error_info = {
                        "task_id": self.task_id,
                        "trace_id": self.trace_id,
                        "error_type": str(init_error),
                        "timestamp": datetime.datetime.now().isoformat()
                    }
                    with open(task_error_path, 'w', encoding='utf-8') as f:
                        json.dump(error_info, f, indent=2, ensure_ascii=False)
                    logger.info(f"[{self.trace_id}] Failed task info saved to: {task_error_path}")
                except Exception as save_error:
                    logger.error(f"[{self.trace_id}] Failed to save failed task info - task_id: {self.task_id}, Error: {save_error}")
                if isinstance(init_error, ResourceExhaustedException):
                    raise init_error
            finally:
                # Set event regardless of success or failure so waiting parties know initialization process is complete
                self.env_ready.set()
            # If initialization failed, throw exception
            if not self.env_init_success:
                raise RuntimeError(f"[{self.trace_id}] Environment initialization failed - task_id: {self.task_id}")
            
            # --- get initial observation as first frame ----
            obs = self.env._get_obs()
            obs_img = obs["screenshot"]
            image_size = Image.open(BytesIO(obs["screenshot"])).size
            frame0 = await storage.save_frame.remote(self.task_root, 0, obs_img)
            self._set_first_frame(obs_img, frame0)
            

            step, done = 0, False
            logger.info(f"[{self.trace_id}] Environment initialization complete, starting main loop - task_id: {self.task_id}")

            # def format_chat(messages):
            #     formatted = ""
            #     for m in messages:
            #         content = m["content"]

            #         # If content is list (multimodal message)
            #         if isinstance(content, list):
            #             # Only take text part
            #             texts = [c["text"] for c in content if c.get("type") == "text"]
            #             content_str = "\n".join(texts)
            #         else:
            #             # Regular string
            #             content_str = content

            #         formatted += f"{content_str}<|im_end|>\n"

            #     return formatted

            # base_messages = format_chat(self.base_messages_for_save)

            # tokenize_response = await self._call_model_tokenize(
            #                 model_pool, base_messages,
            #                 self.runner_cfg.model_pool)

            # # prompt_token_ids = tokenize_response
            # prompt_token_ids = await model_pool.process_text.remote(self.base_messages_for_save)
            # --- main loop ----
            # MAX_T, RETRIES, BACKOFF = 10, 3, 2
            prompt_token_ids=None
            flag_vllm_error = False
            while step < self.max_steps and not done:
                
                if step + 1 == self.max_steps:
                    action = "FAIL"
                    logger.warning(f"[{self.trace_id}] Reached maximum step limit, setting action to FAIL - task_id: {self.task_id}, step: {step}")

                else:
                    st_step = time.time()
                    print("step start time: ", st_step)
                    # build prompt
                    messages = self._build_messages()
                    print("len messages: ", len(messages))

                    # ---- call vLLM serve ----

                    # if local model pool not avaiable, use pre launched model server
                    st = time.time()
                    if model_pool:
                        response, model_path, vllm_logp, token_ids = await self._call_model(
                            model_pool, messages,
                            self.runner_cfg.model_pool, step)
                        
                    model_duration = time.time() - st
                    logger.info(f"[{self.trace_id}] Model response - task_id: {self.task_id}, step: {step}, "
                                f"Duration: {model_duration:.2f}s, response_length: {len(response) if response else 0}")
                    logger.debug(f"[{self.trace_id}] Full model response: {response}")
                    
                    if response is None:
                        action = "FAIL"
                        flag_vllm_error = True
                        logger.warning(f"[{self.trace_id}] Model response is empty, setting action to VLLM ERROR - task_id: {self.task_id}, step: {step}")

                    else:
                        try:
                            low_level_instruction, action_codes = self._parse(response, image_size)
                            action = self._normalize_env_action(action_codes)
                        except Exception as e:
                            logger.warning(f"Action parsing failed! task_id: {self.task_id}, Error: {e}")
                            low_level_instruction, action_codes, action = "", ["FAIL"], "FAIL"
                        self._add_text(response, low_level_instruction, action_codes, step + 1)


                # ---- interact with env ----
                if action in ["DONE", "FAIL"]:
                    logger.info(f"[{self.trace_id}] Task completed - task_id: {self.task_id}, step: {step}, action: {action}")
                
                if action == "VLLM ERROR":
                    logger.error(f"[{self.trace_id}] Model call failed, task terminated - task_id: {self.task_id}, step: {step}")
                    # raise RuntimeError(f"Model call failed, task_id: {self.task_id}, step: {step}")

                # Execute the action
                st = time.time()
                obs, reward, done, info = await self._run_step_async(action)
                obs_img = obs["screenshot"]

                env_duration = time.time() - st
                logger.info(f"[{self.trace_id}] Environment step execution complete - task_id: {self.task_id}, step: {step}, "
                            f"Duration: {env_duration:.2f}s, done: {done}")


                # ---- save screenshot ----
                if action not in ["DONE", "FAIL", "VLLM ERROR"]:
                    frame_path = await storage.save_frame.remote(
                        self.task_root, step + 1, obs_img)
                    
                    self._add_image(obs_img, frame_path)
                
                # ---- save current trajectory
                await storage.save_partial_traj.remote(self.task_root, step + 1, self._build_trajectory())

                # ---- save vllm logp ----
                await storage.save_partial_pt.remote(self.task_root, step + 1, vllm_logp, token_ids, prompt_token_ids)
                await storage.save_parsed_actions.remote(self.task_root, self.parsed_actions)
                
                step_duration = time.time() - st_step
                # self._log_latency(step, model_duration, env_duration, step_duration)
                
                step += 1

            

            # calculate and save reward
            """Evaluate whether the task is successfully completed."""
            try:
                # reward = self.env.evaluate()
                reward = await asyncio.to_thread(self.env.evaluate)
            except Exception as e:
                logger.error(f"Evaluation failed: {e}")
                # Re-throw exception so caller knows evaluation failed
                # TODO regression testing
                reward = -1


            # reward = self.env.evaluate()
            # if action == "VLLM ERROR":
            if flag_vllm_error:
                reward = -2
            await storage.save_reward.remote(self.task_root, reward)
            logger.info(f"[{self.trace_id}] Task evaluation complete - task_id: {self.task_id}, reward: {reward}")
            
            # save trajectory json
            full_messages = self.base_messages_for_save + self.save_dialogue_full
            if full_messages:
                last_msg = full_messages[-1]
                if (
                    last_msg.get("role") == "user" and
                    len(last_msg.get("content", [])) == 1 and
                    last_msg["content"][0].get("type") == "image_url"
                ):
                    full_messages.pop()
                    logger.info(f"[{self.trace_id}] Removing last user message containing only image - task_id: {self.task_id}")
            await storage.save_episode.remote(self.task_root, full_messages)
            await storage.save_parsed_actions.remote(self.task_root, self.parsed_actions)

            await model_pool.save_messages_reward.remote(
                full_messages,
                reward,
                self.task_id,
                self.trace_id,
                self.parsed_actions,
            )

            # When enabled together with PR, release the VM before the slow PR pass.
            # The coordinator will run the deferred post-processing on this actor.
            if (
                bool(getattr(self.process_reward_cfg, "release_env_before_evaluation", False))
                and pr_model_pool
                and reward >= 0
                and step > 0
                and not flag_vllm_error
            ):
                deferred_data = {
                    "full_messages": full_messages,
                    "reward": reward,
                    "step": step,
                    "action": action,
                    "flag_vllm_error": flag_vllm_error,
                    "model_path": model_path,
                    "storage_root": storage_root,
                }
                if self.env is not None:
                    try:
                        self.env.close()
                        logger.info(f"[{self.trace_id}] Environment closed before PR evaluation - task_id: {self.task_id}")
                    except Exception as e:
                        logger.error(f"[{self.trace_id}] Error closing environment before PR evaluation: {e}")
                    finally:
                        self.env = None
                return {"defer_pr": True, "data": deferred_data}

            process_reward_list = []
            detailed_prm_json = None
            if pr_model_pool and reward >= 0 and step > 0 and not flag_vllm_error:
                logger.info(f"[{self.trace_id}] ⏳ Starting Dual-Pool PRM Evaluation...")
                process_reward_list, detailed_prm_json = await self.evaluate_process_reward(
                    pr_model_pool, full_messages, float(reward)
                )
                logger.info(f"[{self.trace_id}] ✅ PRM Evaluation done. Array: {process_reward_list}")
            else:
                # 降级：如果没有传池子，或任务失败报错，塞入长度对齐的默认分数
                process_reward_list = [-1] * (step if action == 'DONE' else step-1)

            if detailed_prm_json:
                try:
                    # 替换为通过 StorageActor 统一异步保存
                    await storage.save_prm_detail.remote(self.task_root, detailed_prm_json)
                    logger.info(f"[{self.trace_id}] Saved detailed PRM JSON via StorageActor.")
                except Exception as e:
                    logger.error(f"[{self.trace_id}] Failed to save detailed PRM JSON: {e}")
            
            # Insert into mysql database
            if self.runner_cfg.write_to_mysql and not flag_vllm_error:
                if flag_vllm_error:
                    reward = -2
                pr_sql_value = json.dumps(process_reward_list)

                meta = {
                'run_id': storage_root,
                'trajectory_id': self.task_root,
                'task_id': self.task_id,
                'trace_id': self.trace_id,
                'reward': reward,
                'process_reward': pr_sql_value,
                'model_version': model_path,
                'instruction': self.task_cfg["instruction"],
                'num_chunks' : step if action == 'DONE' else step-1     # Spliter function disabled, save step count here
                }
                logger.info(f"[{self.trace_id}] Begin to insert mysql")
                if pr_model_pool:
                    if _valid_process_reward(process_reward_list):
                        if reward>0:
                            logger.info(f"[{self.trace_id}] Stored in mySQL database")
                            await mysql_writer.insert_run.remote(meta)
                        elif reward==0:
                            if step >=4:
                                await mysql_writer.insert_run.remote(meta)
                                logger.info(f"[{self.trace_id}] Stored in mySQL database")
                            else:
                                logger.info(f"[{self.trace_id}] Reward is 0, steps less than 4, not stored in mySQL database")
                        else:
                            logger.info(f"[{self.trace_id}] Reward less than 0, not stored in mySQL database")
                    else:
                        logger.info(f"[{self.trace_id}] Process reward list contains invalid scores, not stored in mySQL database")
                else:
                    if reward>0:
                        logger.info(f"[{self.trace_id}] Stored in mySQL database")
                        await mysql_writer.insert_run.remote(meta)
                    elif reward==0:
                        if step >=4:
                            await mysql_writer.insert_run.remote(meta)
                            logger.info(f"[{self.trace_id}] Stored in mySQL database")
                        else:
                            logger.info(f"[{self.trace_id}] Reward is 0, steps less than 4, not stored in mySQL database")
                    else:
                        logger.info(f"[{self.trace_id}] Reward less than 0, not stored in mySQL database")
            logger.info(f"[{self.trace_id}] Task trajectory execution complete - task_id: {self.task_id}, Total steps: {step}")           
 
        except ResourceExhaustedException:
            raise
        except Exception as e:
            logger.error(f"[{self.trace_id}] Task trajectory execution failed - task_id: {self.task_id}, Error: {e}")
            raise
        finally:
            # Cancel environment timeout monitoring task
            if self.env_timeout_task is not None and not self.env_timeout_task.done():
                self.env_timeout_task.cancel()
                try:
                    await self.env_timeout_task
                except asyncio.CancelledError:
                    pass
            # Ensure environment is released regardless of success or failure
            if self.env is not None:
                try:
                    emu_id = _emulator_id_for_log(self.env)
                    self.env.close()
                    logger.info(f"[{self.trace_id}] Environment closed - Environment emulator_id: {emu_id} - task_id: {self.task_id}")
                except Exception as e:
                    logger.error(f"[{self.trace_id}] Error closing environment - task_id: {self.task_id}, Error: {e}")
                finally:
                    self.env = None

        return True

    async def run_pr_postprocess(self, deferred_data, pr_model_pool, storage, mysql_writer):
        """Run PR evaluation and database insertion after the VM is released."""
        full_messages = deferred_data["full_messages"]
        reward = deferred_data["reward"]
        step = deferred_data["step"]
        action = deferred_data["action"]
        flag_vllm_error = deferred_data["flag_vllm_error"]
        model_path = deferred_data["model_path"]
        storage_root = deferred_data["storage_root"]

        try:
            logger.info(f"[{self.trace_id}] Starting deferred PR evaluation - task_id: {self.task_id}")
            process_reward_list, detailed_prm_json = await self.evaluate_process_reward(
                pr_model_pool, full_messages, float(reward)
            )
            logger.info(f"[{self.trace_id}] Deferred PR evaluation done. Array: {process_reward_list}")

            if detailed_prm_json:
                try:
                    await storage.save_prm_detail.remote(self.task_root, detailed_prm_json)
                    logger.info(f"[{self.trace_id}] Saved detailed PR JSON via StorageActor.")
                except Exception as e:
                    logger.error(f"[{self.trace_id}] Failed to save detailed PR JSON: {e}")

            await self._write_mysql_result(
                mysql_writer=mysql_writer,
                pr_model_pool=pr_model_pool,
                storage_root=storage_root,
                process_reward_list=process_reward_list,
                reward=reward,
                step=step,
                action=action,
                flag_vllm_error=flag_vllm_error,
                model_path=model_path,
            )
            return True
        except Exception as e:
            logger.error(f"[{self.trace_id}] Deferred PR post-processing failed: {e}")
            return False

    async def _write_mysql_result(
        self,
        mysql_writer,
        pr_model_pool,
        storage_root,
        process_reward_list,
        reward,
        step,
        action,
        flag_vllm_error,
        model_path,
    ):
        """Write deferred PR results using the existing insertion rules."""
        if not self.runner_cfg.write_to_mysql or flag_vllm_error:
            return

        pr_sql_value = json.dumps(process_reward_list)
        meta = {
            "run_id": storage_root,
            "trajectory_id": self.task_root,
            "task_id": self.task_id,
            "trace_id": self.trace_id,
            "reward": reward,
            "process_reward": pr_sql_value,
            "model_version": model_path,
            "instruction": self.task_cfg["instruction"],
            "num_chunks": step if action == "DONE" else step - 1,
        }

        logger.info(f"[{self.trace_id}] Begin to insert deferred PR result into mysql")
        if pr_model_pool:
            if _valid_process_reward(process_reward_list):
                if reward > 0:
                    await mysql_writer.insert_run.remote(meta)
                    logger.info(f"[{self.trace_id}] Stored deferred PR result in mySQL database")
                elif reward == 0:
                    if step >= 4:
                        await mysql_writer.insert_run.remote(meta)
                        logger.info(f"[{self.trace_id}] Stored deferred PR result in mySQL database")
                    else:
                        logger.info(f"[{self.trace_id}] Reward is 0, steps less than 4, not stored in mySQL database")
                else:
                    logger.info(f"[{self.trace_id}] Reward less than 0, not stored in mySQL database")
            else:
                logger.info(f"[{self.trace_id}] Process reward list contains invalid scores, not stored in mySQL database")
        else:
            if reward > 0:
                await mysql_writer.insert_run.remote(meta)
                logger.info(f"[{self.trace_id}] Stored deferred result in mySQL database")
            elif reward == 0:
                if step >= 4:
                    await mysql_writer.insert_run.remote(meta)
                    logger.info(f"[{self.trace_id}] Stored deferred result in mySQL database")
                else:
                    logger.info(f"[{self.trace_id}] Reward is 0, steps less than 4, not stored in mySQL database")
            else:
                logger.info(f"[{self.trace_id}] Reward less than 0, not stored in mySQL database")
    
    def _init_env(self, env_cfg):  
        # Add retry logic for RemoteDesktopEnv initialization
        max_retries = env_cfg.max_retries
        for attempt in range(max_retries):
            try:
                load_dotenv()
                self.env = DesktopEnv(
                            action_space=env_cfg.action_space,
                            provider_name=env_cfg.provider_name,
                            os_type=env_cfg.os_type,
                        )
                self.env.reset(task_config=self.task_cfg)
                time.sleep(60)
                logger.info(f"[{self.trace_id}] Environment initialization successful - task_id: {self.task_id}, Attempt: {attempt + 1}")
                break
            except Exception as e:
                emu_id = _emulator_id_for_log(self.env)
                logger.warning(
                    f"[{self.trace_id}] Environment initialization failed - task_id: {self.task_id}, "
                    f"Attempt: {attempt + 1}, Error: {e}; emulator_id: {emu_id}"
                )
                try:
                    if self.env is not None:
                        self.env.close()
                except Exception as close_error:
                    logger.warning(f"[{self.trace_id}] Failed to close env after init error: {close_error}")
                finally:
                    self.env = None
                if attempt == max_retries - 1:
                    logger.error(f"[{self.trace_id}] All retry attempts failed - task_id: {self.task_id}")
                    err_s = str(e).lower()
                    if any(key in err_s for key in ("quota", "failed to start emulator", "no available")):
                        raise ResourceExhaustedException(str(e)) from e
                    raise
                logger.info(f"[{self.trace_id}] Preparing to retry - task_id: {self.task_id}, Next attempt: {attempt + 2}/{max_retries}")


    async def _call_model(self, model_pool, messages: str, model_cfg, step):
        """
        Call model; retry RETRIES times on failure, exponential backoff (backoff**attempt) seconds.
        Return response (str) or None.
        """
        timeout = model_cfg.timeout
        retries = model_cfg.retries
        backoff = model_cfg.backoff
        DEFAULT_MODEL_PATH = model_cfg.ckpt_path
        
        for attempt in range(1, retries + 1):
            try:
                request_seed = model_cfg.seed
                if isinstance(request_seed, int):
                    request_seed += attempt - 1
                response, model_path, vllm_logp, token_ids = await asyncio.wait_for(
                    model_pool.generate.remote(messages,
                                               frequency_penalty=model_cfg.frequency_penalty,
                                               temperature=model_cfg.temperature,
                                               top_p=model_cfg.top_p,
                                               max_tokens=model_cfg.max_tokens,
                                               seed=request_seed,
                                               chat_template_kwargs={"enable_thinking": self.qwen3_enable_thinking},
                                               logprobs=model_cfg.logprobs,
                                               return_tokens_as_token_ids=model_cfg.return_tokens_as_token_ids,       
                                               task_id=self.task_id,
                                                trace_id=self.trace_id,
                                                step=step), 
                    timeout=timeout,
             
                )

                format_error = self._qwen3_tool_call_format_error(response)
                if format_error:
                    logger.warning(
                        f"[{self.trace_id}] Invalid Qwen3 tool call - task_id: {self.task_id}, "
                        f"attempt: {attempt}/{retries}, error: {format_error}"
                    )
                    if attempt == retries:
                        return None, model_path, None, None
                    await asyncio.sleep(backoff ** attempt)
                    continue

                return response, model_path, vllm_logp, token_ids
            except (asyncio.TimeoutError, ray.exceptions.RayError) as e:
                if attempt == retries:
                    model_path = await asyncio.wait_for(
                        model_pool.get_last_model_version.remote(),
                        timeout=timeout
                    )
                    if not model_path:
                        logger.error(f" Can't get last model version, use default model path in model_cfg")
                        model_path = DEFAULT_MODEL_PATH
                    logger.error(f"[{self.trace_id}] Model call failed - task_id: {self.task_id}, Retry attempts: {attempt}, return last model version Error: {e}")
                    return None, model_path, None, None
                await asyncio.sleep(backoff ** attempt)   # 2s, 4s, 8s …

    async def _call_model_tokenize(self, model_pool, messages: str, model_cfg):
        """
        Call tokenize interface to tokenize formatted base_messages.
        Return token_ids (List[int]).
        """
        timeout = model_cfg.timeout
        retries = model_cfg.retries
        backoff = model_cfg.backoff
        DEFAULT_MODEL_PATH = model_cfg.ckpt_path

        for attempt in range(1, retries + 1):
            try:
                token_ids = await asyncio.wait_for(
                    model_pool.tokenize.remote(messages),
                    timeout=timeout
                )
                return token_ids
            except (asyncio.TimeoutError, ray.exceptions.RayError) as e:
                if attempt == retries:
                    logger.error(f"[{self.trace_id}] Tokenize call failed - task_id: {self.task_id}, Retry attempts: {attempt}, Error: {e}")
                    return None
                await asyncio.sleep(backoff ** attempt) 

    async def _async_call_prm(self, pr_model_pool, messages, temperature=0.1, max_tokens=2048):
        """调用 PRM 服务并清洗 JSON"""
        retries = 3
        for attempt in range(retries):
            try:
                # 复用 model_pool 的底层通讯！
                prm_result = await asyncio.wait_for(
                    pr_model_pool.generate_vllm.remote(
                        messages=messages,
                        temperature=temperature,
                        max_tokens=max_tokens,
                        chat_template_kwargs={"enable_thinking": False}
                    ),
                    timeout=120.0
                )
                response = prm_result[0] if isinstance(prm_result, (tuple, list)) else prm_result
                if response:
                    # 清洗 JSON
                    content = re.sub(r"```json\s*", "", response, flags=re.IGNORECASE)
                    content = re.sub(r"```\s*", "", content).strip()
                    return content
            except Exception as e:
                if attempt == retries - 1:
                    logger.error(f"[{self.trace_id}] PRM Call failed after retries: {e}")
                    return None
                await asyncio.sleep(5)
        return None

    async def _async_call_seekjudge(self, pr_model_pool, messages, **kwargs):
        """Call the active PR pool while preserving optional tool-call messages."""
        chat_template_kwargs = dict(kwargs.get("chat_template_kwargs") or {})
        chat_template_kwargs["enable_thinking"] = False
        kwargs["chat_template_kwargs"] = chat_template_kwargs
        retries = max(
            1,
            int(getattr(self.process_reward_cfg, "max_retries", 3)),
        )
        for attempt in range(retries):
            try:
                result = await asyncio.wait_for(
                    pr_model_pool.generate_vllm.remote(messages=messages, **kwargs),
                    timeout=1800.0,
                )
                return result[0] if isinstance(result, (tuple, list)) else result
            except Exception as error:
                if attempt == retries - 1:
                    raise RuntimeError("SeekJudge model call failed") from error
                await asyncio.sleep(5)
    
    async def evaluate_process_reward(self, pr_model_pool, full_messages: List[Dict], outcome_reward: float):
        # 1. 提取出 task_instruction 和 steps
        task_instruction, steps = self._extract_trace_from_memory(full_messages)
        total_steps = len(steps)
        step_score_array = [-1] * total_steps
        group_score_array = [-1] * total_steps

        def raw_pr_payload():
            return _raw_process_reward_payload(step_score_array, group_score_array)

        if self.pr_backend == "seekjudge":
            try:
                async def call_seekjudge(messages, **kwargs):
                    return await self._async_call_seekjudge(pr_model_pool, messages, **kwargs)

                return await evaluate_seekjudge_process_reward(
                    full_messages=full_messages,
                    task_instruction=task_instruction or self.task_cfg.get("instruction", ""),
                    source_dir=os.path.join(self.root, self.task_root),
                    async_call_llm_fn=call_seekjudge,
                    config=self.process_reward_cfg,
                    task_id=self.task_id,
                )
            except Exception as error:
                logger.exception(f"[{self.trace_id}] SeekJudge evaluation failed: {error}")
                return raw_pr_payload(), None

        if not steps or not pr_model_pool:
            return raw_pr_payload(), None

        pr_mode = self.pr_mode
        enable_pr_phase3 = self.enable_pr_phase3
        scoring = self.pr_scoring
        group_fn = scoring.group_multi if pr_mode == "multi" else scoring.group_single
        score_fn = scoring.score_multi if pr_mode == "multi" else scoring.score_single
        refine_fn = scoring.refine_multi if pr_mode == "multi" else scoring.refine_single

        # 2. 构造一个能自动绑定当前模型池的 async 回调函数
        async def call_llm_callback(messages, max_tokens=4096):
            return await self._async_call_prm(
                pr_model_pool=pr_model_pool, 
                messages=messages, 
                temperature=0.1, 
                max_tokens=max_tokens
            )

        # 3. ✨ 一行代码搞定复杂的分组逻辑！
        enriched_subgoals = await group_fn(task_instruction, steps, call_llm_callback)
        
        if not enriched_subgoals:
            logger.warning(f"[{self.trace_id}] Grouping returned empty, fallback to -1")
            return raw_pr_payload(), None

        try:
            scoring.validate_grouped_step_coverage(
                enriched_subgoals, total_steps, require_windows=(pr_mode == "multi")
            )
        except scoring.errors as error:
            logger.warning(f"[{self.trace_id}] Invalid grouping structure: {error}")
            return raw_pr_payload(), None

        logger.info(f"[{self.trace_id}] Grouping success, got {len(enriched_subgoals)} parts.")

        previous_group_result_lines = []
        logger.info(f"[{self.trace_id}] Start forward scoring for {len(enriched_subgoals)} groups...")
        
        # 为了兼容图片路径读取，由于我们的 evaluate 是在当前任务完成后立刻触发的
        # storage 上的文件其实就在你的 self.task_root 对应的目录下 (或者绝对路径)
        # 这里你可以直接传入你保存图片的根目录，如果不确定可以写 "./"
        task_dir = os.path.join(self.root, self.task_root)

        for idx in range(len(enriched_subgoals)):
            previous_group_results = "\n".join(previous_group_result_lines)
            eval_res = await score_fn(
                task_instruction=task_instruction,
                task_root_path=task_dir,
                all_subgoals=enriched_subgoals,
                current_idx=idx,
                previous_group_results=previous_group_results,
                async_call_llm_fn=call_llm_callback
            )
            
            if not eval_res:
                logger.warning(f"[{self.trace_id}] Eval for Group {idx+1} failed after structural retries. Rejecting trace.")
                return raw_pr_payload(), None
            
            # Store scoring results first; final process rewards are computed after all groups are scored.
            enriched_subgoals[idx]['evaluation'] = eval_res

            group_conclusion = eval_res.get("current_group_conclusion", {})
            try:
                group_score = scoring.group_score(group_conclusion)
            except scoring.errors as error:
                logger.warning(f"[{self.trace_id}] Invalid Phase 2 group judgment: {error}")
                return raw_pr_payload(), None

            previous_group_result_lines.append(
                scoring.previous_group_line(
                    idx, enriched_subgoals[idx], group_conclusion, group_score
                )
            )

        phase3_refinement = None
        if enable_pr_phase3:
            logger.info(f"[{self.trace_id}] Start Phase 3 group-level refinement...")
            phase3_refinement = await refine_fn(
                task_instruction=task_instruction,
                all_subgoals=enriched_subgoals,
                async_call_llm_fn=call_llm_callback
            )
            if phase3_refinement:
                for item in phase3_refinement.get("refined_group_conclusions", []):
                    if not isinstance(item, dict):
                        continue
                    try:
                        group_index = int(item.get("group_index", 0)) - 1
                    except Exception:
                        continue
                    if 0 <= group_index < len(enriched_subgoals):
                        enriched_subgoals[group_index]["refined_group_conclusion"] = item
                logger.info(f"[{self.trace_id}] Phase 3 refinement done.")
            else:
                logger.warning(f"[{self.trace_id}] Phase 3 failed after structural retries. Rejecting trace.")
                return raw_pr_payload(), None

        try:
            scoring.validate_final_step_coverage(enriched_subgoals, total_steps)
        except scoring.errors as error:
            logger.warning(f"[{self.trace_id}] Incomplete final scoring structure: {error}")
            return raw_pr_payload(), None

        refined_score_by_group = {}
        if phase3_refinement:
            for item in phase3_refinement.get("refined_group_conclusions", []):
                if not isinstance(item, dict):
                    continue
                try:
                    group_index = int(item.get("group_index", 0)) - 1
                    refined_score = scoring.refined_score(item)
                except scoring.errors:
                    continue
                if 0 <= group_index < len(enriched_subgoals) and refined_score >= 0:
                    refined_score_by_group[group_index] = refined_score

        for idx, subgoal in enumerate(enriched_subgoals):
            eval_res = subgoal.get("evaluation")
            if not isinstance(eval_res, dict):
                continue
            group_conclusion = eval_res.get("current_group_conclusion", {})
            try:
                group_score = refined_score_by_group.get(
                    idx,
                    scoring.group_score(group_conclusion),
                )
            except scoring.errors:
                group_score = -1
            if group_score < 0:
                continue

            for sc in eval_res.get("step_critic", []):
                if not isinstance(sc, dict):
                    continue
                try:
                    s_idx = int(sc.get("step_index", 1)) - 1
                    step_score = float(sc.get("score", -1))
                except Exception:
                    continue
                if 0 <= s_idx < total_steps and step_score >= 0:
                    step_score_array[s_idx] = step_score
                    group_score_array[s_idx] = group_score

        if (
            len(step_score_array) != total_steps
            or len(group_score_array) != total_steps
            or any(score < 0 for score in step_score_array)
            or any(score < 0 for score in group_score_array)
        ):
            logger.warning(f"[{self.trace_id}] Process rewards do not cover every step. Rejecting trace.")
            return raw_pr_payload(), None

        detailed_prm_data = {
            "task_id": self.task_id,
            "trace_id": self.trace_id,
            "overall_reward": outcome_reward,
            "scoring_version": scoring.name,
            "task_instruction": task_instruction,
            "enable_pr_phase3": enable_pr_phase3,
            "phase3_refinement": phase3_refinement,
            "subgoals": enriched_subgoals  # 包含了步骤、图文、以及刚才塞进去的 evaluation
        }

        return raw_pr_payload(), detailed_prm_data

    async def _run_step_async(self, action):
        loop = asyncio.get_running_loop()
        return await loop.run_in_executor(None, partial(self.env.step, action))

    def _set_first_frame(self, obs_img, frame0):
        processed_b64, processed_size = self._process_qwen_image(obs_img)
        self.processed_screen_size = processed_size
        self.screenshots.append(processed_b64)
        self.screenshot_paths.append(frame0)

        system_prompt = self._build_qwen3_system_prompt(*processed_size)
        instruction_prompt = self._build_qwen3_user_prompt(self.clean_instruction, "None")
        self.base_messages = [
            {"role": "system", "content": [{"type": "text", "text": system_prompt}]},
            {
                "role": "user",
                "content": [
                    {"type": "image_url", "image_url": {"url": "data:image/png;base64," + processed_b64}},
                    {"type": "text", "text": instruction_prompt},
                ],
            },
        ]
        self.base_messages_for_save = [
            {"role": "system", "content": [{"type": "text", "text": system_prompt}]},
            {
                "role": "user",
                "content": [
                    {"type": "image_url", "image_url": frame0},
                    {"type": "text", "text": instruction_prompt},
                ],
            },
        ]

    def _process_qwen_image(self, img_bytes: bytes) -> Tuple[str, Tuple[int, int]]:
        image = Image.open(BytesIO(img_bytes))
        width, height = image.size
        resized_height, resized_width = smart_resize(
            height,
            width,
            factor=32,
            max_pixels=16 * 16 * 4 * 12800,
        )
        image = image.resize((resized_width, resized_height))
        buffer = BytesIO()
        image.save(buffer, format="PNG")
        return base64.b64encode(buffer.getvalue()).decode("utf-8"), (resized_width, resized_height)

    def _build_qwen3_base_messages(self, instruction: str) -> List[Dict]:
        system_prompt = self._build_qwen3_system_prompt(*self.processed_screen_size)
        return [
            {"role": "system", "content": [{"type": "text", "text": system_prompt}]},
            {"role": "user", "content": [{"type": "text", "text": self._build_qwen3_user_prompt(instruction, "None")}]},
        ]

    def _build_qwen3_system_prompt(self, processed_width: int = 1000, processed_height: int = 1000) -> str:
        tools_def = {
            "type": "function",
            "function": {
                "name": "computer_use",
                "description": self._build_qwen3_description_prompt(processed_width, processed_height),
                "parameters": {
                    "type": "object",
                    "required": ["action"],
                    "properties": {
                        "action": {
                            "type": "string",
                            "description": self._qwen3_internal_action_description(),
                            "enum": [
                                "key",
                                "key_down",
                                "key_up",
                                "left_mouse_down",
                                "left_mouse_up",
                                "type",
                                "mouse_move",
                                "left_click",
                                "left_click_drag",
                                "right_click",
                                "middle_click",
                                "double_click",
                                "triple_click",
                                "scroll",
                                "hscroll",
                                "screenshot",
                                "wait",
                                "terminate",
                                "call_user",
                            ],
                        },
                        "keys": {"type": "array", "description": "Required only by `action=key`, `action=key_down`, or `action=key_up`."},
                        "text": {"type": "string", "description": "Required only by `action=type` and `action=call_user`."},
                        "coordinate": {"type": "array", "description": "(x, y) coordinates. Required only by `action=mouse_move` and `action=left_click_drag`, optional for `action=left_mouse_down` and `action=left_mouse_up`."},
                        "pixels": {"type": "number", "description": "Scroll amount. Required only by `action=scroll` or `action=hscroll`."},
                        "time": {"type": "number", "description": "Seconds to wait. Required only by `action=wait`."},
                        "status": {"type": "string", "description": "Task status for terminate.", "enum": ["success", "failure"]},
                    },
                },
            },
        }
        return (
            "You are a multi-purpose intelligent assistant. Based on my requests, you can use tools to help me complete various tasks.\n\n"
            "# Tools\n\n"
            "You have access to the following functions:\n\n"
            "<tools>\n"
            + json.dumps(tools_def, ensure_ascii=False)
            + "\n</tools>\n\n"
            "If you choose to call a function ONLY reply in the following format with NO suffix:\n\n"
            "<tool_call>\n"
            "<function=example_function_name>\n"
            "<parameter=example_parameter_1>\n"
            "value_1\n"
            "</parameter>\n"
            "<parameter=example_parameter_2>\n"
            "This is the value for the second parameter\n"
            "that can span\n"
            "multiple lines\n"
            "</parameter>\n"
            "</function>\n"
            "</tool_call>\n\n"
            "<IMPORTANT>\n"
            "Reminder:\n"
            "- Function calls MUST follow the specified format: an inner <function=...></function> block must be nested within <tool_call></tool_call> XML tags\n"
            "- Required parameters MUST be specified\n"
            "- You may provide optional reasoning for your function call in natural language BEFORE the function call, but NOT after\n"
            "- If there is no function call available, answer the question like normal with your current knowledge and do not tell the user about function calls\n"
            f"- The current date is {datetime.datetime.today().strftime('%A, %B %d, %Y')}.\n"
            f"- Collapsed screenshots appear as text: {self.qwen3_collapse_text}\n"
            "</IMPORTANT>\n\n"
            "# Response format\n\n"
            "For normal UI interaction steps:\n"
            "1) Action: a short imperative describing what to do in the UI.\n"
            "2) A single <tool_call>...</tool_call> block.\n\n"
            "For terminal steps, you may either:\n"
            "- output a final natural-language response with no tool call, or\n"
            "- use a terminal tool call such as call_user or terminate.\n\n"
            "Rules:\n"
            "- For non-terminal UI steps, output exactly in the order: Action, <tool_call>.\n"
            "- Be brief: one sentence for Action.\n"
            "- Do not output anything after a tool call.\n"
            "- Use call_user when you need user information or confirmation.\n"
            "- Use terminate when you want to explicitly end the task with a success or failure status.\n"
            "- If the task is infeasible, say so explicitly in the response."
        )

    def _build_qwen3_description_prompt(self, processed_width: int, processed_height: int) -> str:
        resolution = (
            f"* The screen's resolution is {processed_width}x{processed_height}."
            if getattr(self.runner_cfg, "qwen3_coordinate_type", "relative") == "absolute"
            else "* The screen's resolution is 1000x1000."
        )
        return "\n".join([
            "Use a mouse and keyboard to interact with a computer, and take screenshots.",
            "* This is an interface to a desktop GUI. You do not have access to a terminal or applications menu. You must click on desktop icons to start applications.",
            "* Some applications may take time to start or process actions, so you may need to wait and take successive screenshots to see the results of your actions.",
            resolution,
            "* Whenever you intend to move the cursor to click on an element like an icon, you should consult a screenshot to determine the coordinates of the element before moving the cursor.",
            "* If you tried clicking on a program or link but it failed to load, even after waiting, try adjusting your cursor position so that the tip of the cursor visually falls on the element that you want to click.",
            "* Make sure to click any buttons, links, icons, etc with the cursor tip in the center of the element. Don't click boxes on their edges unless asked.",
        ])

    def _qwen3_internal_action_description(self) -> str:
        return """
* `key`: Performs key down presses on the arguments passed in order, then performs key releases in reverse order.
* `key_down`: Press and hold a single key without releasing it.
* `key_up`: Release a previously held single key.
* `left_mouse_down`: Press and hold the left mouse button.
* `left_mouse_up`: Release the left mouse button.
* `type`: Type a string of text on the keyboard.
* `mouse_move`: Move the cursor to a specified (x, y) pixel coordinate on the screen.
* `left_click`: Click the left mouse button.
* `left_click_drag`: Click and drag the cursor to a specified (x, y) pixel coordinate on the screen.
* `right_click`: Click the right mouse button.
* `middle_click`: Click the middle mouse button.
* `double_click`: Double-click the left mouse button.
* `triple_click`: Triple-click the left mouse button.
* `scroll`: Performs a scroll of the mouse scroll wheel.
* `hscroll`: Performs a horizontal scroll.
* `screenshot`: Capture a new screenshot of the current screen.
* `wait`: Wait specified seconds for the change to happen.
* `terminate`: Terminate the current task and report its completion status.
* `call_user`: Ask user for information or confirmation.
""".strip()

    def _build_qwen3_user_prompt(self, instruction: str, previous_actions: str = "None") -> str:
        return (
            "\nPlease generate the next move according to the UI screenshot, instruction and previous actions.\n\n"
            f"Instruction: {instruction}\n\n"
            "Previous actions:\n"
            f"{previous_actions}"
        )

    def _previous_actions_text(self, start_step: int) -> str:
        previous_actions = [
            f"Step {i + 1}: {self.actions[i]}"
            for i in range(0, min(start_step - 1, len(self.actions)))
        ]
        return "\n".join(previous_actions) if previous_actions else "None"

    def _update_folding_state(self, total_screenshots: int) -> int:
        while (total_screenshots - self.folded_prefix_k) > self.qwen3_image_max:
            self.folded_prefix_k += self.qwen3_fold_size
        if self.folded_prefix_k > total_screenshots:
            self.folded_prefix_k = total_screenshots
        return self.folded_prefix_k

    def _wrap_tool_response(self, parts: List[Dict]) -> List[Dict]:
        return (
            [{"type": "text", "text": "<tool_response>\n"}]
            + parts
            + [{"type": "text", "text": "\n</tool_response>"}]
        )

    def _ensure_empty_think_prefix(self, response: str) -> str:
        text = response or ""
        if re.match(r"^\s*<think>.*?</think>\s*", text, re.DOTALL):
            return text
        return "<think>\n\n</think>\n\n" + text.lstrip("\n")

    def _build_messages(self) -> List[Dict]:
        total_steps = len(self.screenshots)
        self._update_folding_state(total_steps)
        start_step = max(1, total_steps - self.qwen3_history_n)
        previous_actions = self._previous_actions_text(start_step)
        system_prompt = self._build_qwen3_system_prompt(*self.processed_screen_size)
        instruction_prompt = self._build_qwen3_user_prompt(self.clean_instruction, previous_actions)

        messages = [{"role": "system", "content": [{"type": "text", "text": system_prompt}]}]
        for step_num in range(start_step, total_steps + 1):
            is_first_turn = step_num == start_step
            is_collapsed = step_num <= self.folded_prefix_k
            if is_collapsed:
                if is_first_turn:
                    user_content = [{"type": "text", "text": instruction_prompt}]
                else:
                    user_content = self._wrap_tool_response([{"type": "text", "text": self.qwen3_collapse_text}])
            else:
                img_url = "data:image/png;base64," + self.screenshots[step_num - 1]
                if is_first_turn:
                    user_content = [
                        {"type": "image_url", "image_url": {"url": img_url}},
                        {"type": "text", "text": instruction_prompt},
                    ]
                else:
                    user_content = self._wrap_tool_response([{"type": "image_url", "image_url": {"url": img_url}}])
            messages.append({"role": "user", "content": user_content})
            if step_num <= total_steps - 1 and (step_num - 1) < len(self.responses):
                messages.append({"role": "assistant", "content": [{"type": "text", "text": self._ensure_empty_think_prefix(self.responses[step_num - 1])}]})

        try:
            draft_dir = "./draft/message_cache"
            os.makedirs(draft_dir, exist_ok=True)
            with open(os.path.join(draft_dir, f"qwen_messages_step_{total_steps - 1}.json"), "w", encoding="utf-8") as f:
                json.dump(self._sanitize_messages_for_dump(messages), f, ensure_ascii=False, indent=2)
        except Exception as exc:
            logger.warning(f"[{self.trace_id}] Failed to dump Qwen messages: {exc}")
        return messages

    def _sanitize_messages_for_dump(self, messages: List[Dict]) -> List[Dict]:
        sanitized = []
        for message in messages:
            cloned = {"role": message.get("role"), "content": []}
            for part in message.get("content", []) or []:
                if isinstance(part, dict) and part.get("type") == "image_url":
                    url = ((part.get("image_url") or {}).get("url")) or ""
                    cloned["content"].append({"type": "image_url", "image_url": {"url": url[:40] + "...<omitted>"}} if url.startswith("data:image/") else part)
                else:
                    cloned["content"].append(part)
            sanitized.append(cloned)
        return sanitized

    def _add_image(self, img_bytes: bytes, frame_path: str):
        processed_b64, processed_size = self._process_qwen_image(img_bytes)
        self.processed_screen_size = processed_size
        self.screenshots.append(processed_b64)
        self.screenshot_paths.append(frame_path)
        self.save_dialogue.append({"role": "user", "content": [{"type": "image_url", "image_url": frame_path}]})
        self.save_dialogue_full.append({"role": "user", "content": [{"type": "image_url", "image_url": frame_path}]})

    def _add_text(self, assistant_txt: str, low_level_instruction: str = "", action_codes: Optional[List[str]] = None, step: Optional[int] = None):
        action_codes = action_codes or []
        self.responses.append(assistant_txt)
        self.actions.append(low_level_instruction)
        msg = {"role": "assistant", "content": [{"type": "text", "text": assistant_txt}]}
        self.save_dialogue.append(msg)
        self.save_dialogue_full.append(msg)
        self.parsed_actions.append({
            "step": step if step is not None else len(self.responses),
            "response": assistant_txt,
            "low_level_instruction": low_level_instruction,
            "pyautogui_code": action_codes,
        })

    def _build_trajectory(self) -> List[Dict]:
        return self.base_messages_for_save + self.save_dialogue

    def _parse(self, response, image_size):
        low_level_instruction, action_codes = self._parse_qwen3_internal_response(response, image_size)
        logger.info(f"[{self.trace_id}] Parse action code - task_id: {self.task_id}, action: {action_codes}")
        return low_level_instruction, action_codes

    def _normalize_env_action(self, action_codes: List[str]) -> str:
        if not action_codes:
            return "FAIL"
        if len(action_codes) == 1 and action_codes[0] in {"WAIT", "DONE", "FAIL"}:
            return action_codes[0]
        lines = ["import pyautogui", "import time"]
        lines.extend(code for code in action_codes if code not in {"WAIT", "DONE", "FAIL"})
        return "\n".join(lines)

    def _parse_qwen3_internal_response(self, response: str, image_size: Tuple[int, int]) -> Tuple[str, List[str]]:
        format_error = self._qwen3_tool_call_format_error(response)
        if format_error:
            raise ValueError(f"Invalid Qwen3 tool call: {format_error}")

        low_level_instruction = self._extract_action_line(response)
        action_codes: List[str] = []
        infeasible = self._looks_infeasible_response(response)
        for params in self._iter_qwen3_tool_call_params(response):
            action = str(params.get("action") or "").strip()
            if not action:
                continue
            action_codes.extend(self._qwen3_params_to_pyautogui(action, params, image_size, infeasible))
        if not action_codes:
            action_codes.append("FAIL" if infeasible else "DONE")
        if not low_level_instruction:
            low_level_instruction = "Need user input" if action_codes[0] == "FAIL" else self._instruction_from_first_code(action_codes[0])
        return low_level_instruction, action_codes

    def _qwen3_tool_call_format_error(self, response: str) -> Optional[str]:
        text = response or ""
        open_count = len(re.findall(r"<tool_call>", text))
        close_count = len(re.findall(r"</tool_call>", text))

        if open_count == 0:
            return None
        if open_count != close_count:
            return f"unclosed tool_call block ({open_count} open, {close_count} closed)"

        blocks = re.findall(r"<tool_call>(.*?)</tool_call>", text, re.DOTALL)
        if len(blocks) != open_count:
            return "tool_call blocks are not well formed"

        coordinate_actions = {
            "left_click",
            "right_click",
            "middle_click",
            "double_click",
            "triple_click",
            "mouse_move",
            "left_click_drag",
        }
        supported_actions = {
            "key",
            "key_down",
            "key_up",
            "left_mouse_down",
            "left_mouse_up",
            "type",
            "mouse_move",
            "left_click",
            "left_click_drag",
            "right_click",
            "middle_click",
            "double_click",
            "triple_click",
            "scroll",
            "hscroll",
            "screenshot",
            "wait",
            "terminate",
            "call_user",
        }

        for index, block in enumerate(blocks, start=1):
            params = self._parse_xml_tool_call(block)
            if not params:
                return f"tool_call {index} is not a computer_use call or has no parameters"

            action = str(params.get("action") or "").strip()
            if action not in supported_actions:
                return f"tool_call {index} has unsupported action: {action!r}"
            if action in coordinate_actions and self._parse_coordinate(params.get("coordinate")) is None:
                return f"tool_call {index} action {action!r} requires a valid coordinate"
            if action == "type" and "text" not in params:
                return f"tool_call {index} action 'type' requires text"
            if action in {"key", "key_down", "key_up"} and not self._parse_keys(params.get("keys", [])):
                return f"tool_call {index} action {action!r} requires keys"
            if action in {"scroll", "hscroll"} and "pixels" not in params:
                return f"tool_call {index} action {action!r} requires pixels"
            if action == "terminate" and str(params.get("status") or "").strip().lower() not in {"success", "failure"}:
                return "tool_call action 'terminate' requires status success or failure"

        return None

    def _iter_qwen3_tool_call_params(self, response: str):
        for match in re.finditer(r"<tool_call>(.*?)</tool_call>", response or "", re.DOTALL):
            params = self._parse_xml_tool_call(match.group(1))
            if params:
                yield params

    def _parse_xml_tool_call(self, xml_content: str) -> Optional[Dict[str, Any]]:
        params: Dict[str, Any] = {}
        func_match = re.search(r"<function=([^>]+)>", xml_content)
        if not func_match or func_match.group(1) != "computer_use":
            return None
        for match in re.finditer(r"<parameter=([^>]+)>\s*(.*?)\s*</parameter>", xml_content, re.DOTALL):
            name = match.group(1)
            value = match.group(2).strip()
            if value.startswith("[") or value.startswith("{"):
                try:
                    params[name] = json.loads(value)
                    continue
                except json.JSONDecodeError:
                    pass
            params[name] = value
        return params

    def _extract_action_line(self, response: str) -> str:
        for line in (response or "").split("\n"):
            stripped = line.strip()
            if stripped.lower().startswith("action:"):
                return stripped.split(":", 1)[-1].strip()
        return ""

    def _qwen3_params_to_pyautogui(self, action: str, params: Dict[str, Any], image_size: Tuple[int, int], infeasible: bool) -> List[str]:
        coordinate = self._parse_coordinate(params.get("coordinate"))
        codes: List[str] = []
        if action in {"left_click", "right_click", "middle_click", "double_click", "triple_click", "mouse_move"}:
            x, y = self._adjust_qwen3_coordinate(coordinate, image_size) if coordinate else (None, None)
            fn = {"left_click": "click", "right_click": "rightClick", "middle_click": "middleClick", "double_click": "doubleClick", "triple_click": "tripleClick", "mouse_move": "moveTo"}[action]
            codes.append(f"pyautogui.{fn}({x}, {y})" if coordinate else f"pyautogui.{fn}()")
        elif action == "left_click_drag":
            x, y = self._adjust_qwen3_coordinate(coordinate, image_size) if coordinate else (0, 0)
            duration = self._parse_number(params.get("duration", 0.5), 0.5)
            codes.append(f"pyautogui.dragTo({x}, {y}, duration={duration})")
        elif action == "type":
            text = str(params.get("text", "")).replace("\r\n", "\n").replace("\r", "\n")
            chunks = text.split("\n")
            for idx, chunk in enumerate(chunks):
                if chunk:
                    codes.append(f"pyautogui.typewrite({py_string(chunk)})")
                if idx < len(chunks) - 1:
                    codes.append(f"pyautogui.press({py_string('enter')})")
        elif action == "key":
            keys = self._parse_keys(params.get("keys", []), lowercase=True)
            if keys:
                keys_str = ", ".join(py_string(key) for key in keys)
                codes.append(f"pyautogui.hotkey({keys_str})" if len(keys) > 1 else f"pyautogui.press({keys_str})")
        elif action == "key_down":
            codes.extend(f"pyautogui.keyDown({py_string(key)})" for key in self._parse_keys(params.get("keys", []), lowercase=True))
        elif action == "key_up":
            codes.extend(f"pyautogui.keyUp({py_string(key)})" for key in self._parse_keys(params.get("keys", []), lowercase=True))
        elif action == "scroll":
            codes.append(f"pyautogui.scroll({int(self._parse_number(params.get('pixels', 0), 0))})")
        elif action == "hscroll":
            codes.append(f"pyautogui.hscroll({int(self._parse_number(params.get('pixels', 0), 0))})")
        elif action in {"wait", "screenshot"}:
            codes.append("WAIT")
        elif action == "terminate":
            status = str(params.get("status", "success")).strip().lower()
            codes.append("FAIL" if status in {"fail", "failed", "failure", "error", "infeasible"} else "DONE")
        elif action == "call_user":
            codes.append("FAIL" if infeasible else "DONE")
        elif action == "left_mouse_down":
            if coordinate:
                x, y = self._adjust_qwen3_coordinate(coordinate, image_size)
                codes.append(f"pyautogui.moveTo({x}, {y})")
            codes.append("pyautogui.mouseDown(button='left')")
        elif action == "left_mouse_up":
            if coordinate:
                x, y = self._adjust_qwen3_coordinate(coordinate, image_size)
                codes.append(f"pyautogui.moveTo({x}, {y})")
            codes.append("pyautogui.mouseUp(button='left')")
        return codes

    def _adjust_qwen3_coordinate(self, coordinate, image_size: Tuple[int, int]) -> Tuple[int, int]:
        x, y = coordinate
        original_width, original_height = image_size
        if getattr(self.runner_cfg, "qwen3_coordinate_type", "relative") == "absolute":
            processed_width, processed_height = self.processed_screen_size
            return int(float(x) * original_width / processed_width), int(float(y) * original_height / processed_height)
        return int(float(x) * original_width / 999), int(float(y) * original_height / 999)

    def _parse_coordinate(self, raw_coord):
        if isinstance(raw_coord, str):
            try:
                raw_coord = json.loads(raw_coord)
            except Exception:
                return None
        if isinstance(raw_coord, list) and len(raw_coord) >= 2:
            return raw_coord[0], raw_coord[1]
        return None

    def _parse_number(self, raw_value, default=0.0) -> float:
        try:
            return float(raw_value)
        except Exception:
            return float(default)

    def _parse_keys(self, raw_keys, lowercase: bool = False) -> List[str]:
        if isinstance(raw_keys, str):
            try:
                raw_keys = json.loads(raw_keys)
            except Exception:
                try:
                    raw_keys = ast.literal_eval(raw_keys)
                except Exception:
                    pass

        def clean_key_token(key: object) -> str:
            token = str(key).strip()
            token = token.strip(" \t\r\n[](){}\"'")
            token = token.rstrip(" \t\r\n]")
            token = token.lstrip(" \t\r\n[")
            return token.strip()

        def flatten(keys_obj) -> List[str]:
            if keys_obj is None:
                return []
            if isinstance(keys_obj, list):
                values: List[str] = []
                for item in keys_obj:
                    values.extend(flatten(item))
                return values
            values = []
            for part in re.split(r"\s*\+\s*", str(keys_obj).strip()):
                cleaned = clean_key_token(part)
                if cleaned:
                    values.append(cleaned)
            return values

        keys = flatten(raw_keys)
        return [key.lower() for key in keys] if lowercase else keys

    def _looks_infeasible_response(self, text: str) -> bool:
        lowered = (text or "").lower()
        if "[infeasible]" in lowered or "infeasible" in lowered:
            return True

        literal_patterns = [
            "not possible",
            "impossible",
            "not feasible",
            "cannot be completed",
            "can't be completed",
            "cannot be done",
            "cannot complete",
            "can't complete",
            "unable to complete",
            "cannot do this task",
            "can't do this task",
            "cannot complete this task as described",
            "cannot be completed as specified",
            "can't be completed as specified",
            "not available in your country",
            "not available",
            "unavailable",
            "not supported",
            "does not support",
            "doesn't support",
            "cannot natively",
            "does not have a built-in",
            "doesn't have a built-in",
            "does not include",
            "is not among the natively built-in",
            "will fall back to english",
            "requires the official",
            "no bluetooth found",
            "plug in a dongle",
            "folder is empty",
            "downloads folder is empty",
            "do not have the credentials",
            "don't have the credentials",
            "do not have the account credentials",
            "don't have the account credentials",
            "need the user's google account credentials",
            "requires a language pack extension",
            "requires email verification",
            "requires a sign-up",
            "requires sign-up",
            "requires google account credentials",
            "requires a google account",
            "sign in to the google account",
            "drm-protected",
            "drm protection",
            "cannot directly play",
            "no legitimate way",
            "requires a plugin",
            "requires an extension",
            "requires extension",
            "requires plugin",
            "requires a valid account",
            "requires purchase",
            "requires a purchased",
            "no valid account",
            "hidden audio",
            "could you clarify",
        ]
        if any(pattern in lowered for pattern in literal_patterns):
            return True

        regex_patterns = [
            r"\bthere is no [a-z0-9 _-]+\b",
            r"\bno [a-z0-9 _-]+ in [a-z0-9 _-]+ list\b",
            r"\brequires? (an? )?(extension|plugin|account|credentials|hardware|language pack)\b",
            r"\bneed(?:s)? (an? )?(extension|plugin|account|credentials|hardware|language pack)\b",
            r"\b(without|no) (extensions?|plugins?|terminal|ffmpeg|other apps?).{0,120}\\b(cannot|can't|not possible|not feasible)\b",
        ]
        return any(re.search(pattern, lowered) for pattern in regex_patterns)

    def _instruction_from_first_code(self, first_code: str) -> str:
        if first_code == "DONE":
            return "Task completed"
        if first_code == "WAIT":
            return "Waiting"
        if "." in first_code:
            return f"Performing {first_code.split('.', 1)[1].split('(', 1)[0]} action"
        return "Performing action"
        
    def _log_latency(self, step, model_duration, env_duration, step_duration, out_path="timings.jsonl"):
        record = {
            "ts": datetime.datetime.now().isoformat(),
            "task_id": self.task_id,
            "trace_id": self.trace_id,
            "step": step,
            "model_duration": round(model_duration, 4),
            "env_duration": round(env_duration, 4),
            "step_duration": round(step_duration, 4),
        }
        pathlib.Path(out_path).parent.mkdir(exist_ok=True, parents=True)
        with open(out_path, "a") as f:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")

            
    def _remove_dialogue_msg(self, idx: int):
        """
        Delete the idx-th dialogue message in prompt/save,
        and synchronously correct the msg_idx of dialogue sources in image_refs.
        """
        self.prompt_dialogue.pop(idx)
        self.save_dialogue.pop(idx)

        # Update image_refs
        self.image_refs = [
            r if not (r["source"] == "dialogue" and r["msg_idx"] == idx)
            else None                           # Discard the same image reference being deleted
            for r in self.image_refs
        ]
        self.image_refs = [
            (
                {**r, "msg_idx": r["msg_idx"] - 1}
                if r and r["source"] == "dialogue" and r["msg_idx"] > idx # Image indices after idx all -1
                else r
            )
            for r in self.image_refs
            if r                                 # Remove None
        ]

    def _extract_trace_from_memory(self, full_messages: List[Dict]):
        """从内存中提取指令和步骤 (不落盘)"""
        task_instruction = ""
        steps = []
        step_count = 0
        current_image_file = None 

        for msg in full_messages:
            role = msg.get('role')
            content = msg.get('content')

            if role == 'user' and isinstance(content, list):
                for item in content:
                    if item.get('type') == 'text':
                        text = item.get('text', '')
                        if "## User Instruction" in text:
                            match = re.search(r"## User Instruction\s*(.*)", text, re.DOTALL)
                            if match:
                                task_instruction = match.group(1).strip()
                        elif "Instruction:" in text:
                            match = re.search(r"Instruction:\s*(.*?)(?:\n\s*Previous actions:|\Z)", text, re.DOTALL)
                            if match:
                                task_instruction = match.group(1).strip()
                        elif text.startswith("Task:"):
                            task_instruction = text.split("Task:", 1)[1].strip()
                    if item.get('type') == 'image_url':
                        url_part = item.get('image_url', '')
                        current_image_file = url_part if isinstance(url_part, str) else url_part.get('url', '')

            if role == 'assistant' and isinstance(content, list):
                for item in content:
                    if item.get('type') == 'text':
                        step_count += 1
                        steps.append({
                            "step_id": step_count,
                            "raw_content": item.get('text', '').strip(),
                            "image_file": current_image_file
                        })
        return task_instruction, steps
