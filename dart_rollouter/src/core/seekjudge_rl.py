"""SeekJudge-style process-reward evaluation integrated into the RL runtime."""

from __future__ import annotations

import asyncio
import base64
import copy
import io
import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Awaitable, Callable, Dict, List, Optional

from PIL import Image, ImageDraw

from src.core.seekjudge_holistic import HOLISTIC_SYSTEM_PROMPT
from src.utils.ui_tars_utils import (
    parse_action_to_structure_output,
    parsing_response_to_pyautogui_code,
)


LABEL_SCORE_MAP = {
    "milestone": 1.0,
    "correct_decision": 1.0,
    "error_correction": 1.0,
    "neutral": 1.0,
    "off_path": 1.0,
    "missed_correction": 1.0,
    "meaningless": 0.0,
    "false_claim": 0.0,
    "wrong_action": 0.0,
}
OVERALL_KEYS = {
    "goal_infeasibility",
    "goal_ambiguity",
    "exploration_degree",
    "path_redundancy",
    "positive_evidence_reliability",
    "negative_evidence_reliability",
    "completion_progress",
}
STEP_LABELS = frozenset(LABEL_SCORE_MAP)
FORMAT_RETRIES = 5
_TERMINAL_PATTERNS = (
    "report_infeasible",
    "send_msg_to_user",
    "computer.terminate(status",
    "finished(",
)

_BOX_TOKEN_RE = re.compile(r"<\|box_start\|>|<\|box_end\|>")
_IMAGE_NAME_RE = re.compile(r"^(.*?)(\d+)(\.[^.]+)$")
_POINT_RE = re.compile(
    r"pyautogui\.(?:click|rightClick|middleClick|doubleClick|moveTo|dragTo)"
    r"\(\s*(?:x\s*=\s*)?([0-9.]+)\s*,\s*(?:y\s*=\s*)?([0-9.]+)"
)
_ANSWER_OVERALL_RE = re.compile(
    r"<answer_overall>\s*(\{.*?\})\s*</answer_overall>", re.DOTALL
)
_ANSWER_STEP_RE = re.compile(r"<answer_step>\s*(.*?)\s*</answer_step>", re.DOTALL)

_PROMPT_DIR = Path(__file__).with_name("seekjudge_prompts")
OVERALL_PROMPT = (_PROMPT_DIR / "overall_prompt.txt").read_text(encoding="utf-8")
STEP_PROMPT = (_PROMPT_DIR / "step_prompt.txt").read_text(encoding="utf-8")

GROUND_PROMPT = """You are a specialized agent for replacing numeric element IDs in UI actions with natural language descriptions.

You will receive:
1. An image (IMG1) with its A11y (accessibility tree) as complementary representation. In the A11y tree, each element is prefixed by a bracketed numeric ID. To identify an element, find its ID and read the description that follows. For example, given `[998] Element A\t\t[999] Element B`, ID 998 refers to Element A and ID 999 refers to Element B.
2. An ACTION that was performed on this image

Your task: Replace ONLY the numeric parameters (element IDs, coordinates) in the action with a precise description of the target UI element. All other parameters must remain UNCHANGED. Output ONLY the function call -- no explanations, no commentary, no code blocks.

**Rules:**
- Preserve the function name EXACTLY (click, type, fill, longpress, scroll, swipe, goto, etc.). Strip any prefix like `pyautogui.` -- output bare function names only.
- Replace numeric element IDs/coordinates with a description of the target UI element. The description serves as a PURE-TEXT substitute for the screenshot -- a reader who CANNOT see the image must understand exactly what was interacted with. Therefore:
  1. Semantic function: State what the element DOES, not just its label. If the label is ambiguous (e.g., an icon labeled "Align Center" that could mean horizontal or vertical), clarify its actual function (e.g., "Align Center Vertically" vs "Align Center Horizontally").
  2. Visual appearance: Describe the icon/element appearance briefly (e.g., "icon with four horizontal lines centered between vertical borders").
  3. Toolbar/section location: State which toolbar, menu group, or UI section the element belongs to (e.g., "in the formatting toolbar", "in the Font section of the ribbon"). Do NOT use meaningless absolute positions like "above column D" or "below the Data menu".
  4. Relative neighbors: Mention immediately adjacent elements on the same toolbar/group (e.g., "between the 'Align Left' and 'Align Right' buttons").
- Keep ALL non-numeric parameters (text strings, directions, URLs, etc.) UNCHANGED.
- Output format: function_name[description, other_params...]
- If the action contains NO numeric parameters at all (e.g., goto with a URL), output it EXACTLY as-is without any modification or explanation.
- Output ONLY the single function call line. Nothing else.

**Bad Output Examples:**
- "Clicked the Submit button" (prose -- not function call format)
- click[button] (too vague -- which button?)
- click['Align Center' button in the formatting toolbar] (ambiguous -- horizontal or vertical center? missing semantic function)
- click['Align Top' button icon, located below the 'Data' menu item] (meaningless absolute position; missing semantic function -- what does "Align Top" do here?)
- pyautogui.click['Align Center' button ...] (wrong -- strip the `pyautogui.` prefix)
- type[search field, NEW TEXT] (changed the text parameter)
- The action `goto(...)` is a navigation command... (explanation -- just output the action)

**Good Output Examples:**
- click['Align Center Horizontally' button (icon: four centered horizontal lines) in the alignment section of the formatting toolbar, between the 'Align Left' and 'Align Right' buttons]
- click['Align Center Vertically' button (icon: horizontal lines centered between top and bottom borders) in the alignment section of the formatting toolbar, between the 'Align Top' and 'Align Bottom' buttons]
- click['Dining chairs' listing link, the 3rd item in the furniture listings on page 3, priced at $20.00, between the 'Bathroom vanity' and 'Marley Silver Bed' entries]
- click['>' next-page link in the pagination controls at the bottom of the listings, after the '4' page number link]
- type[search input field in the top navigation bar next to the magnifying glass icon, original_text_value]
- goto('https://example.com/page')
"""

GROUND_VL_PROMPT = """You are a specialized agent for replacing numeric coordinates in UI actions with natural language descriptions.

You will receive:
1. An image (IMG1) with a RED CIRCLE drawn on it, indicating the exact location where an action was performed
2. An ACTION that was performed at the red circle's location

Your task: Replace ONLY the coordinate parameters (x=..., y=...) in the action with a precise description of the UI element at the red circle. All other parameters must remain UNCHANGED. Output in the original function call format.

**Rules:**
- Preserve the function name EXACTLY (click, type, fill, longpress, scroll, swipe, goto, etc.). Strip any prefix like `pyautogui.` -- output bare function names only.
- Replace numeric element IDs/coordinates with a description of the target UI element. The description serves as a PURE-TEXT substitute for the screenshot -- a reader who CANNOT see the image must understand exactly what was interacted with. Therefore:
  1. Semantic function: State what the element DOES, not just its label. If the label is ambiguous (e.g., an icon labeled "Align Center" that could mean horizontal or vertical), clarify its actual function (e.g., "Align Center Vertically" vs "Align Center Horizontally").
  2. Visual appearance: Describe the icon/element appearance briefly (e.g., "icon with four horizontal lines centered between vertical borders").
  3. Toolbar/section location: State which toolbar, menu group, or UI section the element belongs to (e.g., "in the formatting toolbar", "in the Font section of the ribbon"). Do NOT use meaningless absolute positions like "above column D" or "below the Data menu".
  4. Relative neighbors: Mention immediately adjacent elements on the same toolbar/group (e.g., "between the 'Align Left' and 'Align Right' buttons").
- Keep ALL non-numeric parameters (text strings, directions, URLs, etc.) UNCHANGED.
- Output format: function_name[description, other_params...]
- If the action contains NO numeric parameters at all (e.g., goto with a URL), output it EXACTLY as-is without any modification or explanation.
- Output ONLY the single function call line. Nothing else.

**Bad Output Examples:**
- "Clicked the Submit button" (prose -- not function call format)
- click[button] (too vague -- which button?)
- click['Align Center' button in the formatting toolbar] (ambiguous -- horizontal or vertical center? missing semantic function)
- click['Align Top' button icon, located below the 'Data' menu item] (meaningless absolute position; missing semantic function -- what does "Align Top" do here?)
- pyautogui.click['Align Center' button ...] (wrong -- strip the `pyautogui.` prefix)
- type[search field, NEW TEXT] (changed the text parameter)
- The action `goto(...)` is a navigation command... (explanation -- just output the action)

**Good Output Examples:**
- click['Align Center Horizontally' button (icon: four centered horizontal lines) in the alignment section of the formatting toolbar, between the 'Align Left' and 'Align Right' buttons]
- click['Align Center Vertically' button (icon: horizontal lines centered between top and bottom borders) in the alignment section of the formatting toolbar, between the 'Align Top' and 'Align Bottom' buttons]
- click['Dining chairs' listing link, the 3rd item in the furniture listings on page 3, priced at $20.00, between the 'Bathroom vanity' and 'Marley Silver Bed' entries]
- click['>' next-page link in the pagination controls at the bottom of the listings, after the '4' page number link]
- type[search input field in the top navigation bar next to the magnifying glass icon, original_text_value]
- goto('https://example.com/page')
"""

CONDENSE_PROMPT = """You are a specialized agent for summarizing visual state changes between two consecutive observations.

Each image comes with its A11y tree as complementary evidence. Treat both as a unified observation.

**Task:** Write a summary of WHAT CHANGED from previous observation to current observation. Length should be proportional to the amount of change: if the transition involves major changes (new page, many elements changed), write ~180 tokens; if minimal change (e.g., only a text field updated), write ~30-50 tokens. Max 200 tokens.

**Core Rules:**
- EVIDENCE ONLY: Report only what is directly observable. No speculation, inference, or guessing.
- DUAL-SOURCE IF GIVEN A11y tree: Image and A11y tree are complementary; either may contain information the other lacks. Data present in EITHER source is valid and must be reported. ALWAYS cross-check both sources -- if a search box's text is truncated in the image but A11y shows the full string, report the full A11y version; if the image shows a visual element not in A11y, report it too. If text appears truncated/clipped in one source, use the other to recover the full version.
- CHANGES FIRST: Focus on what is different. Only mention unchanged elements if they provide important context (e.g., confirming which page you're on).
- Do NOT exhaustively list unchanged elements. A brief "rest of page unchanged" suffices.
- HIGH INFORMATION DENSITY: Every sentence should convey new information. No filler.
- DESCRIPTIVE LANGUAGE: Describe observable state differences, not inferred user intent. Say "text changed to X", "element appeared/disappeared", "panel now shows Y". Do NOT say "user navigated to", "page loaded", "system updated" -- these are inferences about causation, not observations.
- DATA CHANGES: For tables/lists, describe what rows/columns appeared, disappeared, or changed. For pagination/counts, only state values that are directly displayed on screen.
- NO COMPUTATION: You are a restater, not a calculator. NEVER perform any arithmetic or logical derivation. This includes but is not limited to: counting items yourself, adding/subtracting values, interpreting date strings into relative time, computing percentages, deriving totals, or any other operation that produces a value not directly present as text in the image (eg OCR) or A11y tree. Only report numbers and values that are explicitly displayed in the source. If a count or total is not directly readable, do NOT produce one.
- CONCISE ENUMERATION: When listing items that share the same format (eg rows in a file list, entries in a table), list only the distinguishing identifiers (eg names). Do NOT repeat per-item details (eg size, date) for every entry. If columns are few, name them once; if many, describe the column types once. The goal is a compact summary, not a data dump.

**Format:** Single continuous paragraph WITHOUT line breaks, markdown, bullets, or headers.

**Structure:** Start with the main change (what area changed and how), then add key details (specific text, positions). Skip unchanged elements unless critical for context.

**Truncated Observations:** If "[truncated]" appears, describe what you CAN see, then append: "After [what was truncated] is truncated, there might be more new [content type] about [general topic]." Never guess truncated content.
"""

ANALYSE_PROMPT = """You are an information retrieval agent. Extract precise factual details from visual and structural observations (images and Accessibility Trees) based on the user's query.

Requirements:
- EXHAUSTIVE: Scan the ENTIRE observation top-to-bottom for ALL items matching the query. Missing a matching item is a critical failure.
- EVIDENCE ONLY: Only report what is directly observable. No speculation, inference, or guessing.
- DUAL-SOURCE: Treat image and A11y tree as complementary evidence. Data present in either source is valid.
- If the A11y tree ends with "[truncated]", state which sections are cut off, but do NOT guess truncated content.
- If nothing relevant is found, state so clearly.
"""

ANALYSE_TOOL = [{
    "type": "function",
    "function": {
        "name": "AnalyseAgent",
        "description": "Retrieves detailed information from a specific image by analyzing its visual content and A11y tree",
        "parameters": {
            "type": "object",
            "properties": {
                "query": {
                    "type": "string",
                    "description": "A descriptive query about what information to retrieve",
                },
                "img_id": {
                    "type": "integer",
                    "description": "The index of the image to analyze (0-based)",
                },
            },
            "required": ["query", "img_id"],
        },
    },
}]

LLMCall = Callable[..., Awaitable[Any]]


@dataclass
class SeekJudgeTrajectory:
    goal: str
    image_paths: List[Path]
    actions: List[str]
    reasonings: List[str]
    raw_responses: List[str]
    warnings: List[str]

    @property
    def step_count(self) -> int:
        return len(self.actions)


def _cfg(config: Any, key: str, default: Any) -> Any:
    if config is None:
        return default
    if isinstance(config, dict):
        return config.get(key, default)
    return getattr(config, key, default)


def _text_content(message: Dict[str, Any]) -> str:
    content = message.get("content", "")
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        return "\n".join(
            item.get("text", "")
            for item in content
            if isinstance(item, dict) and item.get("type") == "text"
        )
    return str(content)


def _image_ref(message: Dict[str, Any]) -> Optional[str]:
    content = message.get("content", [])
    if not isinstance(content, list):
        return None
    for item in content:
        if not isinstance(item, dict) or item.get("type") != "image_url":
            continue
        ref = item.get("image_url", "")
        if isinstance(ref, dict):
            ref = ref.get("url", "")
        return ref if isinstance(ref, str) else None
    return None


def _resolve_image(ref: Optional[str], source_dir: Path) -> Optional[Path]:
    if not ref or ref.startswith("data:image/"):
        return None
    path = Path(ref)
    candidates = [path] if path.is_absolute() else [source_dir / path, source_dir / path.name]
    for candidate in candidates:
        if candidate.is_file():
            return candidate.resolve()
    return None


def _next_numbered_image(current: Optional[Path], source_dir: Path) -> Optional[Path]:
    if current is None:
        return None
    match = _IMAGE_NAME_RE.match(current.name)
    if not match:
        return None
    prefix, number, suffix = match.groups()
    candidate = source_dir / f"{prefix}{int(number) + 1:0{len(number)}d}{suffix}"
    return candidate.resolve() if candidate.is_file() else None


def _parse_pyautogui_action(response: str, image_path: Path) -> str:
    with Image.open(image_path) as image:
        width, height = image.size
    structured = parse_action_to_structure_output(
        _BOX_TOKEN_RE.sub("", response),
        factor=1000,
        origin_resized_height=height,
        origin_resized_width=width,
        model_type="qwen25vl",
        max_pixels=16384 * 28 * 28,
        min_pixels=100 * 28 * 28,
    )
    return parsing_response_to_pyautogui_code(
        structured,
        image_height=height,
        image_width=width,
        input_swap=False,
    )


def _load_saved_actions(source_dir: Path) -> Dict[int, str]:
    path = source_dir / "parsed_actions.json"
    if not path.is_file():
        return {}
    try:
        records = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}
    saved: Dict[int, str] = {}
    for position, record in enumerate(records, start=1):
        if not isinstance(record, dict):
            continue
        try:
            step = int(record.get("step", position))
        except (TypeError, ValueError):
            continue
        codes = record.get("pyautogui_code")
        if isinstance(codes, str) and codes.strip():
            saved[step] = codes.strip()
        elif isinstance(codes, list):
            normalized = [str(code).strip() for code in codes if str(code).strip()]
            if len(normalized) == 1 and normalized[0] in {"WAIT", "DONE", "FAIL"}:
                saved[step] = normalized[0]
            elif normalized:
                saved[step] = "\n".join(["import pyautogui", "import time", *normalized])
    return saved


def extract_seekjudge_trajectory(
    full_messages: List[Dict[str, Any]],
    task_instruction: str,
    source_dir: str | Path,
) -> SeekJudgeTrajectory:
    """Align saved RL messages into N actions and N+1 existing image paths."""
    source_dir = Path(source_dir).resolve()
    warnings: List[str] = []
    transitions: List[Dict[str, Any]] = []
    current_image: Optional[Path] = None

    for message in full_messages:
        role = message.get("role")
        if role == "user":
            resolved = _resolve_image(_image_ref(message), source_dir)
            if resolved is not None:
                current_image = resolved
                if transitions and transitions[-1].get("next_image") is None:
                    transitions[-1]["next_image"] = resolved
        elif role == "assistant":
            response = _text_content(message).strip()
            if not response:
                continue
            if current_image is None:
                warnings.append(f"step {len(transitions) + 1}: missing pre-action screenshot")
                continue
            transitions.append({
                "response": response,
                "image": current_image,
                "next_image": None,
            })

    if not transitions:
        raise ValueError("RL trajectory contains no assistant steps with screenshots")

    for index, transition in enumerate(transitions, start=1):
        next_image = transition["next_image"] or _next_numbered_image(
            transition["image"], source_dir
        )
        if next_image is None:
            next_image = transition["image"]
            warnings.append(f"step {index}: final screenshot missing; previous image path reused")
        transition["next_image"] = next_image

    saved_actions = _load_saved_actions(source_dir)
    actions: List[str] = []
    reasonings: List[str] = []
    raw_responses: List[str] = []
    for index, transition in enumerate(transitions, start=1):
        response = transition["response"]
        action = saved_actions.get(index)
        if action is None:
            try:
                action = _parse_pyautogui_action(response, transition["image"])
            except Exception as error:
                warnings.append(f"step {index}: action parser fallback: {error}")
                action = response
        thought = response.split("Action:", 1)[0].strip() if "Action:" in response else response
        actions.append(action)
        reasonings.append(thought)
        raw_responses.append(response)

    image_paths = [transitions[0]["image"]]
    image_paths.extend(transition["next_image"] for transition in transitions)
    return SeekJudgeTrajectory(
        goal=task_instruction,
        image_paths=image_paths,
        actions=actions,
        reasonings=reasonings,
        raw_responses=raw_responses,
        warnings=warnings,
    )


def _data_uri(path: Path) -> str:
    suffix = path.suffix.lower()
    mime = "image/jpeg" if suffix in {".jpg", ".jpeg"} else f"image/{suffix.lstrip('.') or 'png'}"
    encoded = base64.b64encode(path.read_bytes()).decode("ascii")
    return f"data:{mime};base64,{encoded}"


def _image_item(path: Path) -> Dict[str, Any]:
    return {"type": "image_url", "image_url": {"url": _data_uri(path)}}


def _is_terminal_action(action: str) -> bool:
    stripped = action.strip()
    return stripped in {"DONE", "FAIL"} or any(
        pattern in action for pattern in _TERMINAL_PATTERNS
    )


def _actor_claim(actions: List[str]) -> str:
    for action in reversed(actions[-2:]):
        stripped = action.strip()
        if stripped == "DONE":
            return "The trajectory reached the goal."
        if stripped == "FAIL":
            return "The trajectory failed to reach the goal."
        if any(pattern in action for pattern in _TERMINAL_PATTERNS):
            return action
    return "The trajectory reached the goal."


def _action_points(path: Path, action: str) -> List[tuple[int, int]]:
    with Image.open(path) as source:
        width, height = source.size
    points = []
    for match in _POINT_RE.finditer(action):
        x, y = float(match.group(1)), float(match.group(2))
        if 0 <= x <= 1 and 0 <= y <= 1:
            x, y = x * width, y * height
        points.append((round(x), round(y)))
    return list(dict.fromkeys(points))


def _annotated_image_item(path: Path, points: List[tuple[int, int]]) -> Dict[str, Any]:
    with Image.open(path) as source:
        image = source.convert("RGB")
    width, height = image.size
    draw = ImageDraw.Draw(image)
    radius = max(10, min(width, height) // 50)
    line_width = max(3, radius // 4)
    for x, y in points:
        draw.ellipse(
            [x - radius, y - radius, x + radius, y + radius],
            outline="red",
            width=line_width,
        )
    buffer = io.BytesIO()
    image.save(buffer, format="PNG")
    encoded = base64.b64encode(buffer.getvalue()).decode("ascii")
    return {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{encoded}"}}


async def _preprocess_transition(
    trajectory: SeekJudgeTrajectory,
    index: int,
    call_llm: LLMCall,
    semaphore: asyncio.Semaphore,
) -> str:
    action = trajectory.actions[index]
    before = trajectory.image_paths[index]
    after = trajectory.image_paths[index + 1]
    points = _action_points(before, action)
    terminal = _is_terminal_action(action)

    if points:
        ground_messages = [
            {"role": "system", "content": GROUND_VL_PROMPT},
            {"role": "user", "content": [
                {"type": "text", "text": "Here is the observation with a RED CIRCLE marking the action location:\n\nIMG1:"},
                _annotated_image_item(before, points),
                {"type": "text", "text": f"ACTION performed at the red circle location:\n{action}\n\nPlease identify the UI element at the red circle and output in function call format."},
            ]},
        ]
    else:
        ground_messages = [
            {"role": "system", "content": GROUND_PROMPT},
            {"role": "user", "content": [
                {"type": "text", "text": "Here is the observation to analyze:\n\nIMG1:"},
                _image_item(before),
                {"type": "text", "text": "A11y tree: Not Given."},
                {"type": "text", "text": f"ACTION to be performed on IMG1:\n{action}\n\nPlease identify what UI element is being interacted with."},
            ]},
        ]

    before_item = _annotated_image_item(before, points) if points else _image_item(before)
    location_note = (
        [{"type": "text", "text": "The red circle in IMG1 marks where the action was performed. Focus on describing what changed in that area."}]
        if points else []
    )
    condense_messages = [
        {"role": "system", "content": CONDENSE_PROMPT},
        {"role": "user", "content": [
            {"type": "text", "text": "Here are the two observations to analyze:\n\nIMG1 (earlier state):"},
            *location_note,
            before_item,
            {"type": "text", "text": "A11y tree: Not Given."},
            {"type": "text", "text": "IMG2 (later state):"},
            _image_item(after),
            {"type": "text", "text": "A11y tree: Not Given."},
            {"type": "text", "text": "Please analyze the visual changes between these two observations."},
        ]},
    ]

    async def limited(messages, max_tokens):
        async with semaphore:
            return await call_llm(messages, temperature=0.2, max_tokens=max_tokens)

    if terminal:
        grounded, changed = "I get my claim", await limited(condense_messages, 250)
    else:
        grounded, changed = await asyncio.gather(
            limited(ground_messages, 128),
            limited(condense_messages, 250),
        )
    return f"ACTION: {str(grounded).strip()}\n\nRESULT: {str(changed).strip()}"


async def _analyse_image(image_path: Path, query: str, call_llm: LLMCall) -> str:
    messages = [
        {"role": "system", "content": ANALYSE_PROMPT},
        {"role": "user", "content": [
            {"type": "text", "text": f"QUERY: {query}"},
            {"type": "text", "text": "Observation data follows (scan ALL of it for matches):"},
            _image_item(image_path),
            {"type": "text", "text": "A11y tree: Not Given."},
            {"type": "text", "text": f"REMINDER -- answer this query using ALL matching data from above.\nQUERY: {query}"},
        ]},
    ]
    result = await call_llm(messages, temperature=0.2, max_tokens=1000)
    return str(result).strip()


def _initial_evidence_messages(
    trajectory: SeekJudgeTrajectory,
    summaries: List[str],
) -> List[Dict[str, Any]]:
    parts = [
        f"GOAL: {trajectory.goal}",
        f"UNVERIFIED_CLAIM: {_actor_claim(trajectory.actions)}",
        f"TRAJECTORY_INFO: {len(trajectory.image_paths)} images indexed 0 to {len(trajectory.image_paths) - 1}.",
        "STATE_SUMMARIES:",
    ]
    for index, summary in enumerate(summaries, start=1):
        parts.append(f"Transition {index} (img{index - 1} -> img{index}):\n{summary}")
    parts.append(
        "Please analyze this trajectory and provide your evaluation following the "
        "format specified in the system prompt (Reasoning and Sub-metrics)."
    )
    return [
        {"role": "system", "content": HOLISTIC_SYSTEM_PROMPT},
        {"role": "user", "content": "\n\n".join(parts)},
    ]


async def _gather_evidence(
    trajectory: SeekJudgeTrajectory,
    summaries: List[str],
    call_llm: LLMCall,
    loop_limit: int,
) -> List[Dict[str, Any]]:
    messages = _initial_evidence_messages(trajectory, summaries)
    for _ in range(loop_limit):
        response = await call_llm(
            messages,
            temperature=0.2,
            max_tokens=8192,
            tools=ANALYSE_TOOL,
            tool_choice="auto",
            return_message=True,
        )
        if isinstance(response, str):
            response = {"role": "assistant", "content": response}
        tool_calls = response.get("tool_calls") or []
        if not tool_calls:
            parsed = _parse_result(str(response.get("content", "")))
            if (
                _valid_overall(parsed.get("answer_overall"))
                and _valid_steps(parsed.get("answer_step"), trajectory.step_count)
            ):
                messages.append(response)
            break
        messages.append(response)

        async def execute(tool_call):
            call_id = tool_call.get("id", "analyse")
            try:
                arguments = json.loads(tool_call["function"]["arguments"])
                img_id = arguments.get("img_id")
                query = str(arguments.get("query", ""))
                if tool_call["function"].get("name") != "AnalyseAgent":
                    raise ValueError("unknown tool")
                if not isinstance(img_id, int) or not 0 <= img_id < len(trajectory.image_paths):
                    raise ValueError(f"img_id must be in [0, {len(trajectory.image_paths) - 1}]")
                result = await _analyse_image(trajectory.image_paths[img_id], query, call_llm)
                content = json.dumps({"analyse": result}, ensure_ascii=False)
            except Exception as error:
                content = json.dumps({"error": str(error)}, ensure_ascii=False)
            return {"role": "tool", "content": content, "tool_call_id": call_id}

        messages.extend(await asyncio.gather(*(execute(call) for call in tool_calls)))
    return messages


def _parse_result(text: str) -> Dict[str, Any]:
    parsed: Dict[str, Any] = {}
    overall_match = _ANSWER_OVERALL_RE.search(text)
    if overall_match:
        try:
            parsed["answer_overall"] = json.loads(overall_match.group(1))
        except json.JSONDecodeError:
            pass
    step_match = _ANSWER_STEP_RE.search(text)
    if step_match:
        labels = []
        for line in step_match.group(1).splitlines():
            line = line.strip()
            if line:
                labels.append(line.split(":", 1)[-1].strip())
        parsed["answer_step"] = labels
    return parsed


def _valid_overall(value: Any) -> bool:
    return isinstance(value, dict) and set(value) == OVERALL_KEYS


def _valid_steps(value: Any, count: int) -> bool:
    return isinstance(value, list) and len(value) == count and all(item in STEP_LABELS for item in value)


async def _run_output_stage(
    context: List[Dict[str, Any]],
    prompt: str,
    call_llm: LLMCall,
    validator: Callable[[Dict[str, Any]], bool],
    max_retries: int = FORMAT_RETRIES,
) -> tuple[Dict[str, Any], str]:
    last = {}
    for _ in range(max_retries):
        messages = copy.deepcopy(context)
        messages.append({"role": "user", "content": prompt})
        response = await call_llm(messages, temperature=0.2, max_tokens=8192)
        response_text = str(response)
        last = _parse_result(response_text)
        if validator(last):
            thought = _ANSWER_OVERALL_RE.sub("", response_text)
            thought = _ANSWER_STEP_RE.sub("", thought).strip()
            return last, thought
    raise ValueError(f"SeekJudge output validation failed after {max_retries} attempts: {last}")


def seekjudge_process_reward(result: Dict[str, Any], step_count: int) -> Dict[str, Any]:
    labels = result.get("answer_step")
    overall = result.get("answer_overall")
    if not _valid_overall(overall):
        raise ValueError("SeekJudge answer_overall is invalid")
    if not _valid_steps(labels, step_count):
        raise ValueError(f"SeekJudge answer_step must contain {step_count} valid labels")
    raw_completion = overall["completion_progress"]
    if isinstance(raw_completion, bool):
        raise ValueError("SeekJudge completion_progress must be numeric")
    try:
        completion_percent = float(raw_completion)
    except (TypeError, ValueError) as error:
        raise ValueError("SeekJudge completion_progress must be numeric") from error
    if not 0 <= completion_percent <= 100:
        raise ValueError("SeekJudge completion_progress must be in [0, 100]")
    completion = completion_percent / 100.0
    return {
        "format": "raw_pr_seekjudge_v1",
        "step_scores": [LABEL_SCORE_MAP[label] for label in labels],
        "group_scores": [1.0] * step_count,
        "trajectory_score": completion,
        "step_labels": list(labels),
    }


async def evaluate_seekjudge_process_reward(
    full_messages: List[Dict[str, Any]],
    task_instruction: str,
    source_dir: str | Path,
    async_call_llm_fn: LLMCall,
    config: Any = None,
    task_id: str = "unknown",
) -> tuple[Dict[str, Any], Dict[str, Any]]:
    trajectory = extract_seekjudge_trajectory(full_messages, task_instruction, source_dir)
    concurrency = max(1, int(_cfg(config, "preprocess_concurrency", 8)))
    semaphore = asyncio.Semaphore(concurrency)
    summaries = await asyncio.gather(*(
        _preprocess_transition(
            trajectory,
            index,
            async_call_llm_fn,
            semaphore,
        )
        for index in range(trajectory.step_count)
    ))
    evidence_messages = await _gather_evidence(
        trajectory,
        list(summaries),
        async_call_llm_fn,
        max(0, int(_cfg(config, "loop_limit", 8))),
    )
    stage_context = evidence_messages[1:]
    overall, overall_thought = await _run_output_stage(
        stage_context,
        OVERALL_PROMPT,
        async_call_llm_fn,
        lambda result: _valid_overall(result.get("answer_overall")),
    )
    num_states = trajectory.step_count + 1
    step_prompt = STEP_PROMPT + f"""

This specific trajectory has {num_states} steps, with step_index from 0 to {num_states - 1}.
Therefore, you MUST output exactly {trajectory.step_count} transition lines inside <answer_step>.

Mapping rule:
- Transition 1 corresponds to the action at step_index=0 and the state change from step_index=0 to step_index=1.
- ....
- Transition {trajectory.step_count} corresponds to the action at step_index={trajectory.step_count - 1} and the state change from step_index={trajectory.step_count - 1} to step_index={trajectory.step_count}.

Do NOT output a separate line for step_index={trajectory.step_count} itself, because it is the final state after the last transition.
    """
    step_context = copy.deepcopy(stage_context)
    if overall_thought:
        step_context.append({"role": "user", "content": overall_thought})
    steps, _ = await _run_output_stage(
        step_context,
        step_prompt,
        async_call_llm_fn,
        lambda result: _valid_steps(result.get("answer_step"), trajectory.step_count),
    )
    native_result = {
        "answer_overall": overall["answer_overall"],
        "answer_step": steps["answer_step"],
    }
    reward = seekjudge_process_reward(native_result, trajectory.step_count)
    detailed = {
        "backend": "seekjudge",
        "task_id": task_id,
        "image_paths": [str(path) for path in trajectory.image_paths],
        "adapter_warnings": trajectory.warnings,
        "state_summaries": list(summaries),
        "native_result": native_result,
        "adapted_process_reward": reward,
    }
    return reward, detailed
