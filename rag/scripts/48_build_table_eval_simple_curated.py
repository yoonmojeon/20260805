"""Build 66 simple, page-blind, single-cell table QA questions for review."""
from __future__ import annotations

import argparse
import importlib
import json
import re
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = ROOT / "scripts"
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

practical = importlib.import_module("47_build_table_eval_practical")

DEFAULT_MANIFEST = ROOT / "data/processed/index/unified_kr_tables_v2/index_manifest.json"
DEFAULT_CHUNKS = ROOT / "data/processed/chunks_v2"
DEFAULT_OUT = ROOT / "data/eval/table_questions_22docs_practical_v1_curated.jsonl"
DEFAULT_REVIEW = ROOT / "data/eval/table_questions_22docs_practical_v1_curated_review.md"

NOISE_RE = re.compile(
    r"\[수식기호\]|세부항목|통상적인\s*정의|(?:^|\s)col(?:umn)?[_ ]?\d+|_\d+$",
    re.I,
)
LONG_NUMBER_RUN_RE = re.compile(r"(?:\d+[ .,/~-]+){5,}\d+")
PAGE_RE = re.compile(r"\d+\s*페이지|\bp\.?\s*\d+\b", re.I)

QUESTION_OVERRIDES = {
    "TC22_001": "RSTH 12·22·23·24 관을 확관한 후 관 끝의 허용 바깥지름은 원래 관 바깥지름의 몇 배인가?",
    "TC22_002": "RD·RY·RW·RU 308L 용접재료 시험에 적용되는 강종은 무엇인가?",
    "TC22_003": "형상이 복잡하거나 한 개의 중량이 10톤을 넘는 주강품은 제품마다 시험재가 몇 개 필요한가?",
    "TC22_004": "기관실 격벽은 어느 위치의 횡격벽을 의미하는가?",
    "TC22_005": "호퍼탱크 경사판과 연결된 이중선측 수평거더 웨브는 어떤 방법으로 평가하는가?",
    "TC22_006": "선저 만곡부 외판은 외판의 어느 부분을 의미하는가?",
    "TC22_007": "좌굴패널은 어떤 판 패널을 의미하는가?",
    "TC22_008": "AC-S·AC-SD·AC-A·AC-T의 판과 국부 지지부재 허용응력은 어느 규정을 적용하는가?",
    "TC22_009": "넘침식 또는 순차식 평형수 교환에는 어떤 설계하중 시나리오를 적용하는가?",
    "TC22_010": "실린더 출구별 배기가스 온도가 평균에서 벗어나면 이중차단·배출밸브의 자동 작동이 요구되는가?",
    "TC22_011": "잔류응력을 측정한 경우 시험 결과 문서에 어떤 정보를 기록해야 하는가?",
    "TC22_012": "적층제조 최종 재료의 제조법 승인에는 지침의 어느 장을 적용하는가?",
    "TC22_013": "재화중량이 10만 톤 초과 15만 톤 이하인 선박의 안전사용하중은 몇 톤인가?",
    "TC22_014": "유리섬유강화재의 인장·굽힘·압축·층간전단·흡수율·유리함량 시험은 어떤 판정기준을 적용하는가?",
    "TC22_015": "구명정 승정구역이나 개방갑판의 임시 안전대피구역에 필요한 방화 보존성 등급은 무엇인가?",
    "TC22_016": "주 운송화물명을 표시하는 Chemical Carrier 특기사항의 설계 규정은 어느 장인가?",
    "TC22_017": "이중선체 Oil/Chemical Tanker의 설계에는 어느 장을 적용하는가?",
    "TC22_018": "ESP·EXP 부호가 있는 Oil/Bulk/Ore Carrier의 설계에는 어느 장을 적용하는가?",
    "TC22_019": "주요 지지부재는 최종강도 검토 대상인가?",
    "TC22_020": "화물창 구역의 최소 용접 다리 길이는 몇 mm인가?",
    "TC22_021": "이중저 늑판은 어떤 구조평가 방법을 적용하는가?",
    "TC22_022": "빌지저장탱크는 제4차 이후 정기검사에서 내부검사 대상인가?",
    "TC22_023": "Bilge System의 CMS 통일명칭은 무엇인가?",
    "TC22_024": "Sea Water Service System의 CMS 통일명칭은 무엇인가?",
    "TC22_025": "평형수탱크·빌지탱크·드레인 저장탱크·체인로커 한쪽 면의 부식추가는 몇 mm인가?",
    "TC22_026": "호퍼탱크 경사판과 연결된 이중선측 수평거더 웨브의 종방향 구조평가 방법은 무엇인가?",
    "TC22_027": "파형격벽 스툴 내부 다이아프램 웨브는 어떤 방법으로 평가하는가?",
    "TC22_028": "선수격벽 뒤에 있는 체인로커의 시험압력수두는 어떻게 정하는가?",
    "TC22_029": "횡·종방향 수밀격벽의 최소 순두께 산정식은 무엇인가?",
    "TC22_030": "단일선측 산적화물선 선창 내 늑골의 피로강도 평가 위치는 어디인가?",
    "TC22_031": "어느 면도 해수에 접하지 않는 부재의 부식 두께는 몇 mm인가?",
    "TC22_032": "인장하중을 받는 보강판 또는 일반 보강재는 어떤 손상모드로 평가하는가?",
    "TC22_033": "셀가이드 설계의 하중조합 1에서 수직방향 하중을 적용하는가?",
    "TC22_034": "1·2구역의 비선수미 격벽에는 외판에서 몇 m 이내까지 판 구조요건을 적용하는가?",
    "TC22_035": "Winterization E3(t)의 외부 설계 대기온도 조건은 무엇인가?",
    "TC22_036": "PC3·PC4·PC5 선박 조타장치의 최소 회전속도는 초당 몇 도인가?",
    "TC22_037": "내항 또는 보호수역 운항 시 허용 정수중 전단력은 어떤 기호로 표시하는가?",
    "TC22_038": "1장 12절의 다섯 번째 그림 번호는 어떻게 표기하는가?",
    "TC22_039": "두께가 15mm 초과 20mm 이하인 부재에 사용하는 강재 급은 무엇인가?",
    "TC22_040": "점화용 연료분사 또는 스파크 점화장치가 오작동하면 이중차단·배출밸브가 자동 작동해야 하는가?",
    "TC22_041": "전기·전자·원격제어 기동장치의 허용 경사각도는 몇 도인가?",
    "TC22_042": "보일러·과열기·재열기의 수압시험 압력은 설계압력의 몇 배인가?",
    "TC22_043": "창구와 맨홀을 2차방벽 또는 방벽간 구역에 설치할 때 어떤 강재를 사용해야 하는가?",
    "TC22_044": "Butane-propane 혼합물의 UN 분류번호는 무엇인가?",
    "TC22_045": "운송온도가 -10°C 미만이고 -55°C 이상인 기본형 탱크는 선체구조를 2차방벽으로 사용할 수 있는가?",
    "TC22_046": "선급이 추가로 요구하는 기관 표시·경보항목은 어디에 표시해야 하는가?",
    "TC22_047": "하역용 집크레인과 갠트리크레인의 충격하중계수는 얼마인가?",
    "TC22_048": "하역용 집크레인과 갠트리크레인의 작업계수는 얼마인가?",
    "TC22_049": "소선지름이 0.20mm 이상 1.00mm 이하일 때 최대지름과 최소지름의 허용 차이는 몇 mm인가?",
    "TC22_050": "B1·B2·B3·B4·B5 선박의 스톡리스 선수앵커 한 개당 질량은 각각 얼마인가?",
    "TC22_051": "평수구역 운항선의 상갑판 아래 일반 장소에는 어떤 형식의 현창을 사용해야 하는가?",
    "TC22_052": "길이 25m 이상 30m 미만인 선박에는 동력 빌지펌프가 몇 대 필요한가?",
    "TC22_053": "1층 선루 전단벽의 안덮개 설치비율은 얼마인가?",
    "TC22_054": "SA0·SA1·SA2·SA3 항해범위 선박의 선측부 최소 해수압력은 얼마인가?",
    "TC22_055": "무할로겐 고등급 에틸렌 프로필렌 고무 절연물의 최고 허용 도체온도는 몇 °C인가?",
    "TC22_056": "정격전압이 1,000V 초과 7,200V 이하인 설비의 최소 시험전압은 얼마인가?",
    "TC22_057": "이중저 샤프트터널 또는 파이프터널에 설치하는 배전반의 보호등급은 무엇인가?",
    "TC22_058": "적층용 액상 수지의 광물 함유량은 제조자 공칭값에서 몇 %까지 벗어날 수 있는가?",
    "TC22_059": "구리 함량이 0.1% 미만인 알루미늄합금 탱크의 최소 공칭 내부식 판두께는 몇 mm인가?",
    "TC22_060": "카야와 베닌 마호가니의 내구성 등급은 무엇인가?",
    "TC22_061": "주갑판 아래의 안덮개 설치비율은 얼마인가?",
    "TC22_062": "SA0 항해범위의 여름철 운항거리 제한은 얼마인가?",
    "TC22_063": "HS17 로프의 절단하중은 몇 kN인가?",
    "TC22_064": "단열재료의 화재 및 화염전파 저항 시험에는 어떤 시험규격을 적용하는가?",
    "TC22_065": "압력조절 실패 보호조치를 적용한 구역 1에는 추가 전기설비 조치가 필요한가?",
    "TC22_066": "멤브레인 탱크에는 어떤 2차방벽이 필요한가?",
}


def compact(value) -> str:
    return practical.compact(value)


def clean_topic(candidate: dict) -> str:
    topic = practical.topic_from(candidate)
    topic = re.sub(r"\s*\((?:계속|continued)\)\s*$", "", topic, flags=re.I).strip()
    topic = re.sub(r"\s+", " ", topic)
    return topic


def simple_candidate(candidate: dict) -> bool:
    row = compact(candidate.get("row_key"))
    column = compact(candidate.get("column"))
    answer = compact(candidate.get("answer"))
    topic = clean_topic(candidate)
    combined = f"{topic} {row} {column} {answer}"
    if NOISE_RE.search(combined) or LONG_NUMBER_RUN_RE.search(row):
        return False
    if topic == "해당 규정" or re.match(r"^\d+(?:\s|$)", topic):
        return False
    if not (3 <= len(topic) <= 46):
        return False
    if not (2 <= len(row) <= 40) or " / " in row or row.count(",") > 3:
        return False
    if not (2 <= len(column) <= 30) or " / " in column:
        return False
    if answer in {"-", "–", "—"} or not (1 <= len(answer) <= 36):
        return False
    if re.fullmatch(r"\(\d+\)", column):
        return False
    if len(re.findall(r"[가-힣A-Za-z]+", row)) == 0:
        return False
    return True


def simple_score(candidate: dict) -> float:
    row = compact(candidate["row_key"])
    column = compact(candidate["column"])
    answer = compact(candidate["answer"])
    topic = clean_topic(candidate)
    score = float(candidate.get("practical_score") or 0.0)
    score += 0.8 if len(row) <= 34 else 0.0
    score += 0.5 if len(column) <= 24 else 0.0
    score += 0.4 if len(answer) <= 24 else 0.0
    score += 0.3 if len(topic) <= 38 else 0.0
    score -= 0.35 * row.count(" / ")
    score -= 0.25 * column.count(" / ")
    return score


def render_question(candidate: dict) -> str:
    topic = clean_topic(candidate)
    row = compact(candidate["row_key"])
    column = compact(candidate["column"])
    topic = re.sub(r"\s*기준\s*$", "", topic).strip()
    question = f"{topic}에서 {row}에 해당하는 {column} 값은 무엇인가?"
    return re.sub(r"\s+", " ", question).strip()


def choose_three(candidates: list[dict], used_questions_global: set[str]) -> list[dict]:
    eligible = [candidate for candidate in candidates if simple_candidate(candidate)]
    eligible.sort(key=lambda c: (-simple_score(c), c["page"], c["table_id"]))
    selected: list[dict] = []
    used_tables: set[str] = set()
    used_pages: set[int] = set()
    used_pairs: set[tuple[str, str]] = set()

    def take(require_new_page: bool) -> None:
        for candidate in eligible:
            pair = (candidate["row_key"], candidate["column"])
            question = render_question(candidate)
            if len(selected) >= 3:
                return
            if candidate["table_id"] in used_tables or pair in used_pairs or question in used_questions_global:
                continue
            if require_new_page and candidate["page"] in used_pages:
                continue
            if len(question) > 105:
                continue
            selected.append(candidate)
            used_tables.add(candidate["table_id"])
            used_pages.add(candidate["page"])
            used_pairs.add(pair)
            used_questions_global.add(question)

    take(True)
    take(False)
    return selected


def build_rows(manifest: dict, chunks_dir: Path) -> tuple[list[dict], list[dict]]:
    rows: list[dict] = []
    audit: list[dict] = []
    used_questions_global: set[str] = set()
    qnum = 1
    for doc_id in manifest.get("doc_ids") or []:
        candidates = practical.enrich_candidates(doc_id, chunks_dir / doc_id / "table_chunks.jsonl")
        selected = choose_three(candidates, used_questions_global)
        audit.append(
            {
                "doc_id": doc_id,
                "eligible": sum(simple_candidate(candidate) for candidate in candidates),
                "selected": len(selected),
                "tables": [candidate["table_id"] for candidate in selected],
            }
        )
        if len(selected) != 3:
            continue
        for candidate in selected:
            row = practical.base_row(
                f"TC22_{qnum:03d}",
                render_question(candidate),
                "single_cell_lookup",
                "open_corpus",
                candidate,
                [practical.gold_cell(candidate)],
            )
            row["generator_version"] = "table_eval_practical_v1_curated"
            row["assistant_curated"] = True
            row["human_verified"] = False
            row["question"] = QUESTION_OVERRIDES.get(row["qid"], row["question"])
            rows.append(row)
            qnum += 1
    return rows, audit


def validate(rows: list[dict]) -> dict:
    return {
        "questions": len(rows),
        "documents": len({row["gold_doc_id"] for row in rows}),
        "page_hints": sum(bool(PAGE_RE.search(row["question"])) for row in rows),
        "pdf_names": sum(".pdf" in row["question"].lower() for row in rows),
        "part_phrases": sum(bool(re.search(r"KR\s*\d+편\s*기준", row["question"])) for row in rows),
        "comparison_questions": sum("비교" in row["question"] for row in rows),
        "formula_noise": sum("[수식기호]" in row["question"] for row in rows),
        "over_125_chars": sum(len(row["question"]) > 125 for row in rows),
        "manual_override_missing": sum(row["qid"] not in QUESTION_OVERRIDES for row in rows),
    }


def write_review(path: Path, rows: list[dict], audit: list[dict]) -> None:
    checks = validate(rows)
    lines = [
        "# KR tables practical QA v1 — assistant-curated simple questions",
        "",
        "> 복잡한 비교·다중조건 문항을 제외한 단일 셀 조회형 평가셋이다. 질문 문장은 정리했지만 모범답안은 구조화 데이터에서 가져왔으므로 최종 운영 승인 전 PDF 원문 대조가 필요하다.",
        "",
        f"- 문항: {checks['questions']}",
        f"- 문서: {checks['documents']}",
        f"- 페이지/파일명/편명 힌트: {checks['page_hints'] + checks['pdf_names'] + checks['part_phrases']}",
        f"- 비교 문항: {checks['comparison_questions']}",
        "",
    ]
    for row in rows:
        lines.extend(
            [
                f"## {row['qid']}",
                "",
                f"- 질문: {row['question']}",
                f"- 모범답안: {row['gold_answer']}",
                f"- 숨은 근거: `{row['gold_file_name']}` p.{row['gold_page']} / `{row['gold_table_id']}`",
                f"- 근거 셀: {row['gold_row_key']} / {row['gold_column']} = {row['gold_answer']}",
                "- [ ] PDF 원문과 정답·단위가 일치한다",
                "",
            ]
        )
    missing = [item for item in audit if item["selected"] != 3]
    if missing:
        lines.extend(["## 생성 누락", "", "```json", json.dumps(missing, ensure_ascii=False, indent=2), "```", ""])
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--chunks-dir", type=Path, default=DEFAULT_CHUNKS)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--review", type=Path, default=DEFAULT_REVIEW)
    args = parser.parse_args()

    manifest = json.loads(args.manifest.read_text(encoding="utf-8"))
    rows, audit = build_rows(manifest, args.chunks_dir)
    practical.write_jsonl(args.out, rows)
    write_review(args.review, rows, audit)
    print(json.dumps(validate(rows), ensure_ascii=False, indent=2))
    print(json.dumps([item for item in audit if item["selected"] != 3], ensure_ascii=False, indent=2))
    print(f"wrote {args.out}")
    print(f"wrote {args.review}")


if __name__ == "__main__":
    main()
