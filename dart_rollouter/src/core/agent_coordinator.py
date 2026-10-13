import ray
import asyncio
import copy
import time
import logging
import uuid
import os
import json
from typing import List, Dict, Set, Optional
from src.utils.log_config import setup_logging
from datetime import datetime


def _load_runner(protocol: str):
    name = str(protocol or "qwen3").strip().lower()
    if name in {"qwen3", "qwen3vl", "qwen"}:
        from src.core.trajectory_runner_qwen3 import TrajectoryRunnerActor, ResourceExhaustedException
    elif name in {"uitars", "ui-tars", "ui_tars", "original"}:
        from src.core.trajectory_runner_uitars import TrajectoryRunnerActor, ResourceExhaustedException
    else:
        raise ValueError(f"Unknown runner.protocol={protocol!r}; expected qwen3 or uitars")
    return TrajectoryRunnerActor, ResourceExhaustedException

# Setup unified logging system
setup_logging()
logger = logging.getLogger(__name__)

class AgentCoordinator:

    def __init__(self, coordinator_cfg):
        self.max_concurrent_envs = coordinator_cfg.max_concurrent_envs  # Maximum concurrent environments
        self.active_tasks: Set[ray.ObjectRef] = set()  # Track running tasks
        self.active_pr_tasks: Set[asyncio.Task] = set()  # Deferred PR tasks, outside VM slots
        self.max_concurrent_pr_jobs = 2
        self.pr_semaphore = asyncio.Semaphore(self.max_concurrent_pr_jobs)
        self.task_queue = asyncio.Queue()  # Remove queue size limit
        self.task_trace_map: Dict[ray.ObjectRef, str] = {}  # Map task reference to trace_id
        self.poll_count = 0  # Track number of times tasks have been fetched
        self.max_task_queue_size = coordinator_cfg.max_task_queue_size  # Maximum number of times to fetch tasks from task_loader
        self.rollout_n = getattr(coordinator_cfg, 'rollout_n', 1)  # Number of repetitions per task, default is 1
        self.dynamic_rollout = getattr(coordinator_cfg, 'dynamic_rollout', False)  # Whether to enable dynamic rollout_n
        self.max_failure_retries = int(getattr(coordinator_cfg, 'max_failure_retries', 8))
        
        # Environment cleanup related
        self.env_cleanup_interval = getattr(coordinator_cfg, 'env_cleanup_interval', 10)  # Environment cleanup interval (seconds)
        self.env_cleanup_timeout = getattr(coordinator_cfg, 'env_cleanup_timeout', 300)  # Environment timeout (seconds, default 5 minutes)
        self.last_env_cleanup_time = 0  # Last environment cleanup time
        self.active_service_ids: Set[str] = set()  # Track active service_ids
        self.task_to_service_map: Dict[ray.ObjectRef, str] = {}  # Map task reference to service_id
        self.task_to_actor_map: Dict[ray.ObjectRef, ray.actor.ActorHandle] = {}  # Map task reference to Actor
        
        # Graceful exit control
        self.should_graceful_exit = False  # Graceful exit flag
        self.exit_reason = ""  # Exit reason
        
        # Statistics
        self.stats = {
            'total_started': 0,
            'total_completed': 0,
            'total_failed': 0,
            'start_time': time.time()
        }
        
        self.RunnerActor = None
        self.ResourceExhaustedException = None
        logger.info(f"AgentCoordinator initialized - Max concurrent environments: {self.max_concurrent_envs}, Max task fetches: {self.max_task_queue_size} times, rollout_n: {self.rollout_n}")
        logger.info(f"Environment cleanup config - Cleanup interval: {self.env_cleanup_interval} seconds, Cleanup timeout: {self.env_cleanup_timeout} seconds")
        
        # Set up asynchronous exception handler to catch unhandled exceptions
        self._setup_exception_handler()

    def _setup_exception_handler(self):
        """Set up asynchronous exception handler"""
        def exception_handler(loop, context):
            exception = context.get('exception')
            if exception:
                # Check if it's a Ray task error
                if "RayTaskError" in str(type(exception)) or "Failed to get screenshot" in str(exception):
                    # Completely silent Ray task errors, as these are already handled in _track_task_completion
                    logger.debug(f"Caught Ray task exception (expected behavior, already handled)")
                else:
                    logger.error(f"Caught unhandled asynchronous exception: {exception}")
                    logger.error(f"Exception context: {context}")
            else:
                logger.error(f"Caught unhandled asynchronous error: {context}")
        
        # Only set if there's no current event loop
        try:
            loop = asyncio.get_running_loop()
            loop.set_exception_handler(exception_handler)
            logger.info("Asynchronous exception handler set")
        except RuntimeError:
            # No running event loop, will be set later
            logger.debug("Cannot set exception handler temporarily, will be set when event loop starts")

    def _generate_trace_id(self) -> str:
        """Generate unique trace_id"""
        return f"trace-{uuid.uuid4().hex[:12]}-{int(time.time())}"
    
    def _update_active_service_ids(self):
        """Update active service_id list"""
        # This method needs to extract service_id from running tasks
        # Since we cannot directly get service_id from Ray object references,
        # we need to record at task startup and remove at task completion
        pass

    async def start_rollout(
        self,
        task_loader,
        runner_cfg,
        model_pool,
        storage,
        mysql_writer,
        pr_model_pool=None,
        process_reward_cfg=None,
    ):
        """Main loop: continuously process tasks"""
        protocol = getattr(runner_cfg, "protocol", "qwen3")
        self.RunnerActor, self.ResourceExhaustedException = _load_runner(protocol)
        logger.info(f"Using runner protocol={protocol} actor={self.RunnerActor}")
        release_env_before_pr = bool(
            getattr(process_reward_cfg, "release_env_before_evaluation", False)
        )
        self.max_concurrent_pr_jobs = max(
            1, int(getattr(process_reward_cfg, "max_concurrent_jobs", 2))
        )
        self.pr_semaphore = asyncio.Semaphore(self.max_concurrent_pr_jobs)
        # Ensure exception handler is set (now event loop is running)
        try:
            loop = asyncio.get_running_loop()
            if not hasattr(loop, '_exception_handler') or loop._exception_handler is None:
                self._setup_exception_handler()
        except Exception as e:
            logger.warning(f"Failed to reset exception handler: {e}")
            
        logger.info("Starting task processing loop")
    
        while True:
            try:
                # Check if graceful exit is needed
                if self.should_graceful_exit:
                    logger.info(f"Detected graceful exit signal: {self.exit_reason}")
                    if self.active_tasks:
                        logger.info(f"Waiting for {len(self.active_tasks)} active tasks to complete...")
                        # Wait for all active tasks to complete
                        try:
                            await asyncio.wait(self.active_tasks, timeout=60.0)  # Wait maximum 60 seconds
                        except asyncio.TimeoutError:
                            logger.warning("Timeout waiting for tasks to complete, force exit")
                    await self._wait_for_pr_tasks()
                    logger.info("Graceful exit completed")
                    break
                
                # 1. Fill task queue (fill up to MAX_TASK_QUEUE_SIZE times)
                if (self.task_queue.empty() and 
                    self.poll_count < self.max_task_queue_size and
                    not self.should_graceful_exit):  # Only fill tasks when not in exit state

                    new_tasks_batch = task_loader.poll_for_tasks()
                    logger.info(f"Got new task batch, size: {len(new_tasks_batch)}")
                    if new_tasks_batch:
                        self.poll_count += 1  # Increase fill count
                        added_count = 0
                        
                        # Sort tasks by success rate first
                        # if self.dynamic_rollout:
                        #     sorted_tasks = task_loader.sort_tasks_by_success_rate(new_tasks_batch)
                        # else:   
                        sorted_tasks = new_tasks_batch
                                                
                        for task in sorted_tasks:
                            task_id = task.get('task_config', {}).get('raw', {}).get('task_id', 'unknown')
                            pending_rollouts = task.get('pending_rollouts')

                            if self.dynamic_rollout:
                                # Get dynamic rollout_n value
                                traj_counts, success_rate, dynamic_rollout_n = task_loader.get_dynamic_rollout_n(task_id, self.rollout_n) # Tasks with success rate above threshold will also be sampled, monitored in real-time, and if success rate falls below threshold, will resample self.rollout_n trajectories
                            else:
                                dynamic_rollout_n = self.rollout_n

                            if pending_rollouts is not None:
                                dynamic_rollout_n = min(dynamic_rollout_n, pending_rollouts)

                            # Create dynamic_rollout_n copies for each task
                            for rollout_idx in range(dynamic_rollout_n):
                                task_with_trace = copy.deepcopy(task)
                                trace_id = self._generate_trace_id()
                                task_with_trace['trace_id'] = trace_id
                                task_with_trace['rollout_idx'] = rollout_idx  # Add rollout index
                                task_with_trace['dynamic_rollout_n'] = dynamic_rollout_n  # Add dynamic rollout_n info
                                
                                await self.task_queue.put(task_with_trace)
                                added_count += 1
                                
                                logger.info(f"[{trace_id}] Adding task to queue - task_id: {task_id}, rollout_idx: {rollout_idx + 1}/{dynamic_rollout_n}")
                        logger.info(f"Completed {self.poll_count}th task filling, added {added_count} tasks, current queue size: {self.task_queue.qsize()}")
                    else:
                        # No new tasks but haven't reached fill count limit, wait and try again
                        if self.poll_count < self.max_task_queue_size:
                            await asyncio.sleep(1)
                            continue

                # 2. Start tasks (up to maximum concurrency)
                started_count = 0
                while (len(self.active_tasks) < self.max_concurrent_envs and 
                       not self.task_queue.empty() and
                       not self.should_graceful_exit):  # Only start new tasks when not in exit state

                    try:
                        task_info = await asyncio.wait_for(self.task_queue.get(), timeout=0.1)
                        trace_id = task_info.get('trace_id', 'unknown_trace')
                        task_id = task_info.get('task_config', {}).get('raw', {}).get('task_id', 'unknown')
                        rollout_idx = task_info.get('rollout_idx', 0)
                        dynamic_rollout_n = task_info.get('dynamic_rollout_n', self.rollout_n)
                            
                        logger.info(f"[{trace_id}] Starting task - task_id: {task_id}, rollout_idx: {rollout_idx + 1}/{dynamic_rollout_n}")
                            
                        try:
                            actor_cpus = 0 if (release_env_before_pr and pr_model_pool) else 0.5
                            actor = self.RunnerActor.options(num_cpus=actor_cpus).remote(
                                task_info, runner_cfg, process_reward_cfg
                            )
                            task_ref = actor.run_episode.remote(model_pool, storage, mysql_writer, pr_model_pool)
                            self.active_tasks.add(task_ref)
                            self.task_trace_map[task_ref] = trace_id
                            self.task_to_actor_map[task_ref] = actor  # Save Actor reference
                            
                            # Asynchronously track task completion and service_id management
                            task_completion = asyncio.create_task(
                                self._track_task_completion(
                                    task_ref,
                                    task_info,
                                    task_id,
                                    trace_id,
                                    rollout_idx,
                                    release_env_before_pr,
                                    model_pool,
                                    storage,
                                    mysql_writer,
                                    pr_model_pool,
                                )
                            )
                            # Get service_id
                            asyncio.create_task(self._track_service_id(task_ref, actor))
                            # Only add exception handling callback to task_completion to avoid duplicate processing
                            task_completion.add_done_callback(self._handle_task_exception)
                           
                            self.stats['total_started'] += 1
                            started_count += 1
                            logger.info(f"[{trace_id}] Task started successfully - task_id: {task_id}, rollout_idx: {rollout_idx + 1}/{dynamic_rollout_n}, Current active tasks: {len(self.active_tasks)}")
                        except ray.exceptions.RayTaskError as e:
                            logger.warning(f"[{trace_id}] Ray task creation failed: {e}")
                        except Exception as e:
                            logger.error(f"[{trace_id}] Failed to start task - task_id: {task_id}, rollout_idx: {rollout_idx + 1}/{dynamic_rollout_n}, Error: {e}")
                            
                    except asyncio.TimeoutError:
                        break  # Queue empty, break loop
                
                if started_count > 0:
                    logger.info(f"Started {started_count} tasks this round")

                # 3. Wait for task completion (using more efficient approach)
                if self.active_tasks:
                    # Wait for at least one task to complete, or timeout
                    try:
                        done, pending = await asyncio.wait(
                            self.active_tasks, 
                            return_when=asyncio.FIRST_COMPLETED,
                            timeout=1.0  # 1 second timeout, avoid infinite wait
                        )
                        
                        # Process completed tasks
                        for task_ref in done:
                            # Note: Don't clean up here, let _track_task_completion handle it
                            # This ensures service_id and other resources are properly cleaned up
                            pass
                            
                    except asyncio.TimeoutError:
                        # Timeout is normal, continue loop
                        pass
                else:
                    # No active tasks, check if all work is done or graceful exit is needed
                    if self.should_graceful_exit:
                        logger.info("Graceful exit process completed, all tasks cleaned up")
                        break
                    elif (self.task_queue.empty() and 
                          self.poll_count >= self.max_task_queue_size):
                        logger.info("All tasks completed, preparing to exit")
                        await self._wait_for_pr_tasks()
                        break
                    
                    await asyncio.sleep(0.1)
                    
                # Periodically print statistics
                if self.stats['total_started'] % 10 == 0 and self.stats['total_started'] > 0:
                    self._log_stats()
                    
            except Exception as e:
                logger.error(f"Error in main loop: {e}")
                # Exit loop on serious error
                logger.error("Serious error occurred, ending task processing")
                break
        
        # Main loop ended, print final statistics
        await self._wait_for_pr_tasks()
        self._log_stats()
        logger.info("Task processing loop ended")
        return True  # Return completion status

    def _unwrap_task_error(self, exc: BaseException) -> BaseException:
        cause = getattr(exc, "cause", None)
        return cause if isinstance(cause, BaseException) else exc

    async def _requeue_failed_task(self, task_info: Dict, error: BaseException, wait_seconds: float = 0.0):
        retries = int(task_info.get("failure_retries", 0))
        task_id = task_info.get("task_config", {}).get("raw", {}).get("task_id", "unknown")
        if retries >= self.max_failure_retries:
            self.stats["total_failed"] += 1
            logger.error(
                f"Giving up after {retries} retries - task_id: {task_id}, Error: {error}"
            )
            return
        retry_info = copy.deepcopy(task_info)
        retry_info["failure_retries"] = retries + 1
        retry_info["trace_id"] = self._generate_trace_id()
        if wait_seconds > 0:
            logger.info(
                f"Waiting {wait_seconds:.0f}s before requeue - task_id: {task_id}, "
                f"retry {retry_info['failure_retries']}/{self.max_failure_retries}"
            )
            await asyncio.sleep(wait_seconds)
        await self.task_queue.put(retry_info)
        logger.info(
            f"[{retry_info['trace_id']}] Requeued failed task - task_id: {task_id}, "
            f"retry {retry_info['failure_retries']}/{self.max_failure_retries}"
        )

    async def _track_task_completion(
        self,
        task_ref: ray.ObjectRef,
        task_info: Dict,
        task_id: str,
        trace_id: str,
        rollout_idx: int,
        release_env_before_pr: bool,
        model_pool,
        storage,
        mysql_writer,
        pr_model_pool,
    ):
        """Track task completion status"""
        actor_for_deferred_pr = None
        try:
            start_time = time.time()
            task_failed = False  # Mark if task failed
            
            try:
                result = await task_ref
                duration = time.time() - start_time
                self.stats['total_completed'] += 1
                logger.info(f"[{trace_id}] Task executed successfully - task_id: {task_id}, rollout_idx: {rollout_idx + 1}/{self.rollout_n}, Duration: {duration:.2f} seconds")

                # In two-stage mode the runner has already released its VM.
                # Keep its actor only while the independent PR post-process runs.
                if (
                    release_env_before_pr
                    and isinstance(result, dict)
                    and result.get("defer_pr")
                ):
                    actor_for_deferred_pr = self.task_to_actor_map.get(task_ref)
                    if actor_for_deferred_pr is not None:
                        self._enqueue_deferred_pr(
                            actor_for_deferred_pr,
                            result["data"],
                            pr_model_pool,
                            storage,
                            mysql_writer,
                            trace_id,
                            task_id,
                        )
                
            except BaseException as e:
                task_failed = True
                duration = time.time() - start_time
                original = self._unwrap_task_error(e)
                err_s = str(original).lower() + str(e).lower()
                no_vm = (
                    isinstance(e, self.ResourceExhaustedException)
                    or isinstance(original, self.ResourceExhaustedException)
                    or "quota" in err_s
                    or "failed to start emulator" in err_s
                )
                if isinstance(e, Exception):
                    error_type = type(original).__name__
                    error_msg = str(original)
                    if no_vm:
                        logger.warning(
                            f"[{trace_id}] No VM available, will requeue instead of exiting - "
                            f"task_id: {task_id}, Duration: {duration:.2f} seconds, Error: {error_msg}"
                        )
                        await self._requeue_failed_task(task_info, original, wait_seconds=20.0)
                    else:
                        logger.error(
                            f"[{trace_id}] Task execution failed - task_id: {task_id}, "
                            f"rollout_idx: {rollout_idx + 1}/{self.rollout_n}, Duration: {duration:.2f} seconds, "
                            f"Error: {error_type}: {error_msg}"
                        )
                        await self._requeue_failed_task(task_info, original)

                    actor = self.task_to_actor_map.get(task_ref)
                    if actor:
                        try:
                            ray.kill(actor)
                            logger.info(f"[{trace_id}] Force terminated failed Actor")
                        except Exception as kill_error:
                            logger.warning(f"[{trace_id}] Failed to terminate Actor: {kill_error}")
                else:
                    logger.error(f"[{trace_id}] Task interrupted by system-level exception - task_id: {task_id}, Duration: {duration:.2f} seconds, Error: {e}")
            
            finally:
                try:
                    # 1. Resource cleanup: Remove from active task set
                    self.active_tasks.discard(task_ref)
                    self.task_trace_map.pop(task_ref, None)
                    actor = self.task_to_actor_map.pop(task_ref, None)  # Clean up Actor reference

                    # The deferred PR coroutine owns the actor in the two-stage case.
                    if (
                        release_env_before_pr
                        and actor is not None
                        and actor_for_deferred_pr is None
                    ):
                        try:
                            ray.kill(actor)
                            logger.info(f"[{trace_id}] Sampling actor released")
                        except Exception as kill_error:
                            logger.warning(f"[{trace_id}] Failed to release sampling actor: {kill_error}")
                    
                    # Remove from active service_id list
                    service_id = self.task_to_service_map.pop(task_ref, None)
                    if service_id:
                        self.active_service_ids.discard(service_id)
                        logger.info(f"[{trace_id}] Removed service_id from active environment list: {service_id}")
                    
                    logger.info(f"[{trace_id}] Task resource cleanup completed, Current active tasks: {len(self.active_tasks)}")
                        
                except Exception as cleanup_error:
                    logger.error(f"[{trace_id}] Error during task cleanup: {cleanup_error}")
                    
        except Exception as outer_error:
            logger.error(f"[{trace_id}] Unexpected error in _track_task_completion method: {outer_error}")
            import traceback
            logger.error(f"[{trace_id}] Error stack:\n{traceback.format_exc()}")
        
        except BaseException as system_error:
            logger.error(f"[{trace_id}] _track_task_completion method interrupted by system-level exception: {system_error}")
        
        # Note: Resource cleanup has been handled in inner finally block

    def _enqueue_deferred_pr(
        self,
        actor,
        deferred_data,
        pr_model_pool,
        storage,
        mysql_writer,
        trace_id: str,
        task_id: str,
    ):
        """Schedule PR work without counting it against environment concurrency."""
        pr_task = asyncio.create_task(
            self._run_deferred_pr(
                actor,
                deferred_data,
                pr_model_pool,
                storage,
                mysql_writer,
                trace_id,
                task_id,
            )
        )
        self.active_pr_tasks.add(pr_task)
        pr_task.add_done_callback(self._handle_pr_task_done)

    async def _run_deferred_pr(
        self,
        actor,
        deferred_data,
        pr_model_pool,
        storage,
        mysql_writer,
        trace_id: str,
        task_id: str,
    ):
        await self.pr_semaphore.acquire()
        try:
            logger.info(f"[{trace_id}] Starting deferred PR task - task_id: {task_id}")
            await actor.run_pr_postprocess.remote(
                deferred_data,
                pr_model_pool,
                storage,
                mysql_writer,
            )
            logger.info(f"[{trace_id}] Deferred PR task completed - task_id: {task_id}")
        except Exception as e:
            logger.error(f"[{trace_id}] Deferred PR task failed - task_id: {task_id}, Error: {e}")
        finally:
            self.pr_semaphore.release()
            try:
                ray.kill(actor)
            except Exception as e:
                logger.warning(f"[{trace_id}] Failed to release deferred PR actor: {e}")

    def _handle_pr_task_done(self, task):
        self.active_pr_tasks.discard(task)
        try:
            task.exception()
        except asyncio.CancelledError:
            pass
        except Exception as e:
            logger.error(f"Deferred PR task callback failed: {e}")

    async def _wait_for_pr_tasks(self):
        if not self.active_pr_tasks:
            return
        logger.info(f"Waiting for {len(self.active_pr_tasks)} deferred PR tasks to complete...")
        await asyncio.gather(*list(self.active_pr_tasks), return_exceptions=True)

    async def _track_service_id(self, task_ref: ray.ObjectRef, actor):
        """Track task's service_id"""
        try:
            # Wait for environment initialization to complete (using event mechanism, maximum wait 180 seconds)
          # Wait for environment initialization to complete (through explicitly defined remote method)
            env_ready = await actor.wait_for_env_ready.remote(timeout=180.0)
            if not env_ready:
                logger.warning("Task environment initialization failed or timed out - May be due to insufficient resources or server error")
                return
            
            # Try to get service_id
            service_id = await actor.get_service_id.remote()
            if service_id:
                self.task_to_service_map[task_ref] = service_id
                self.active_service_ids.add(service_id)
                logger.info(f"Tracked task environment service_id: {service_id}")
            else:
                # service_id being None usually means environment initialization failed (e.g., insufficient resources)
                logger.warning("Failed to get task's service_id - May be due to environment initialization failure (insufficient resources or server error)")
                
        except BaseException as e:
            # Handle all exceptions (including Exception and system-level exceptions)
            if isinstance(e, Exception):
                logger.warning(f"Failed to get task service_id: {e}")
            else:
                logger.error(f"Getting service_id process interrupted by system-level exception: {e}")
    
    def _handle_task_exception(self, task):
        """Handle asynchronous task exceptions"""
        try:
            if task.exception() is not None:
                exception = task.exception()
                # Adjust log level based on exception type, and reduce duplicate logs
                if "RayTaskError" in str(type(exception)) or "Failed to get screenshot" in str(exception):
                    # Ray task failures are expected, only log brief info once
                    logger.error(f"Ray task failed (expected behavior): {type(exception).__name__}")
                else:
                    # Other exceptions use ERROR level
                    logger.error(f"Asynchronous task execution failed: {exception}")
            else:
                # Task completed normally
                logger.debug("Asynchronous task completed normally")
                            
        except Exception as e:
            logger.error(f"Error handling asynchronous task exception: {e}")
            
    def _log_stats(self):
        """Log statistics"""
        runtime = time.time() - self.stats['start_time']
        success_rate = self.stats['total_completed']/max(1, self.stats['total_completed'] + self.stats['total_failed'])*100
        
        logger.info(f"=== Task Statistics === Runtime: {runtime:.1f} seconds, "
                   f"Started: {self.stats['total_started']}, "
                   f"Completed: {self.stats['total_completed']}, "
                   f"Failed: {self.stats['total_failed']}, "
                   f"Currently active: {len(self.active_tasks)}, "
                   f"Queue size: {self.task_queue.qsize()}, "
                   f"Fill count: {self.poll_count}/{self.max_task_queue_size}, "
                   f"Success rate: {success_rate:.1f}%")

    async def shutdown(self):
        """Gracefully shut down coordinator"""
        logger.info("Shutting down AgentCoordinator...")
        
        # Wait for all active tasks to complete
        if self.active_tasks:
            active_traces = [self.task_trace_map.get(task_ref, 'unknown') for task_ref in self.active_tasks]
            logger.info(f"Waiting for {len(self.active_tasks)} active tasks to complete, trace_ids: {active_traces}")
            await asyncio.wait(self.active_tasks, timeout=30.0)  # 30 second timeout
            
        self._log_stats()
        logger.info("AgentCoordinator shut down")
