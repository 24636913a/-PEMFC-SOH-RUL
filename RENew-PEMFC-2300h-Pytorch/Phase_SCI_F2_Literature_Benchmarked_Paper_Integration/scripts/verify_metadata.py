"""Reproducible DOI metadata verification; no experimental data/model access.

Default mode validates saved official Crossref responses offline. --fetch obtains
fresh official metadata and rebuilds verified_metadata.json (no credentials).
The recorded scientific search date is the user-local date, 2026-10-17.
"""
from __future__ import annotations
import argparse
from concurrent.futures import ThreadPoolExecutor
import hashlib
import json
from pathlib import Path
import urllib.parse
import urllib.request

STAGE = Path(__file__).resolve().parents[1]
DATE = "2026-10-17"
PAPERS = [
    ("P1", "10.1016/j.apenergy.2022.118835"),
    ("P2", "10.1016/j.ijhydene.2023.03.316"),
    ("P3", "10.1016/j.ijhydene.2024.06.095"),
    ("P4", "10.1016/j.renene.2025.123019"),
    ("P5", "10.1016/j.ijhydene.2025.02.317"),
    ("P6", "10.1016/j.jpowsour.2026.239941"),
    ("A1", "10.1016/j.ijhydene.2026.153481"),
    ("A2", "10.1016/j.ijhydene.2025.02.302"),
    ("A3", "10.1016/j.apenergy.2025.125318"),
    ("TFT", "10.1016/j.ijforecast.2021.03.012"),
    ("DATA_PAPER", "10.1016/j.ijhydene.2026.156060"),
]
PIIS = {
    "P1": "S0306261922002756", "P2": "S0360319923014611",
    "P3": "S0360319924022948", "P4": "S0960148125006810",
    "P5": "S0360319925008948", "P6": "S0378775326006919",
    "A1": "S0360319926001187", "A2": "S0360319925008808",
    "A3": "S0306261925000480", "TFT": "S0169207021000637",
    "DATA_PAPER": "S0360319926026984",
}
PUBLISHER_ISSUE_DATES = {
    "P1": "2022-05-01", "P2": "2023-08-01", "P3": "2024-07-12",
    "P4": "2025-07", "P5": "2025-07-03", "P6": "2026-06-15",
    "A1": "2026-02-06", "A2": "2025-03-20", "A3": "2025-03-15",
    "TFT": "2021-10/2021-12", "DATA_PAPER": "2026-07-13",
}
ZENODO_IDS = ("20742676", "20715542")

def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()

def dates(message, name):
    parts = message.get(name, {}).get("date-parts", [])
    return "-".join(str(v).zfill(2) if n else str(v)
                    for n, v in enumerate(parts[0])) if parts else None

def fetch_one(item):
    key, doi = item
    url = "https://api.crossref.org/works/" + urllib.parse.quote(doi, safe="")
    request = urllib.request.Request(url, headers={"User-Agent": "RealHyFC-Paper-Metadata-Audit/1.0", "Accept": "application/json"})
    with urllib.request.urlopen(request, timeout=40) as response:
        raw = response.read()
    payload = json.loads(raw)
    if payload.get("status") != "ok" or payload["message"]["DOI"].lower() != doi.lower():
        raise ValueError(f"DOI identity mismatch: {key}")
    path = STAGE / "outputs/evidence_index" / f"CROSSREF_{key}.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(raw)
    return key, url

def fetch_zenodo(record_id):
    """Repository metadata only: never follow a file/download/preview link."""
    url = "https://zenodo.org/api/records/" + record_id
    request = urllib.request.Request(url, headers={"Accept": "application/json"})
    with urllib.request.urlopen(request, timeout=40) as response:
        raw = response.read()
    payload = json.loads(raw)
    assert str(payload["id"]) == record_id
    path = STAGE / "outputs/evidence_index" / f"ZENODO_{record_id}.json"
    path.write_bytes(raw)

def build():
    result = []
    for key, doi in PAPERS:
        path = STAGE / "outputs/evidence_index" / f"CROSSREF_{key}.json"
        message = json.loads(path.read_text(encoding="utf-8"))["message"]
        assert message["DOI"].lower() == doi.lower()
        authors = [" ".join(filter(None, (a.get("given"), a.get("family")))) or a.get("name", "") for a in message.get("author", [])]
        row = {
            "id": key, "doi": doi, "title": message["title"][0],
            "authors": authors, "author_metadata": message.get("author", []),
            "journal": message.get("container-title", [None])[0],
            "volume": message.get("volume"), "issue": message.get("issue"),
            "pages": message.get("page"), "article_number": message.get("article-number"),
            "published_online": dates(message, "published-online"),
            "published_print": dates(message, "published-print"),
            "published": dates(message, "published"), "issued": dates(message, "issued"),
            "publisher_issue_date_observed": PUBLISHER_ISSUE_DATES[key],
            "date_caveat": "Crossref has no published-online date for this record. Publisher issue date is distinct from online-first publication and DOI registration dates.",
            "bibliographic_year": int((dates(message,"published-print") or dates(message,"published"))[:4]),
            "doi_url": "https://doi.org/" + doi,
            "publisher_url": "https://www.sciencedirect.com/science/article/pii/" + PIIS[key],
            "crossref_url": "https://api.crossref.org/works/" + urllib.parse.quote(doi,safe=""),
            "raw_crossref_path": path.relative_to(STAGE).as_posix(),
            "raw_crossref_sha256": sha(path), "access_checked_date": DATE,
            "access_grade": "PUBLISHER_METADATA_ABSTRACT_CONFIRMED",
            "full_text_access": "No publicly retrievable full text independently established in this metadata pass; local full-text status is separately documented by the content reviewer.",
            "public_full_text_links": [],
            "crossref_text_mining_links_not_proof_of_open_access": message.get("link", []),
            "license_metadata_not_proof_of_access": message.get("license", []),
            "limitations": "Bibliographic metadata only. No inference about absent experiments from abstract or Crossref fields.",
        }
        if key == "TFT":
            row["public_full_text_links"] = ["https://www.sciencedirect.com/science/article/pii/S0169207021000637", "https://arxiv.org/abs/1912.09363"]
            row["access_grade"] = "PUBLISHER_OPEN_ACCESS_CONFIRMED"
            row["full_text_access"] = "Publisher labels open access and Creative Commons; author preprint is publicly available on arXiv. Content read extent is separately documented."
        elif key == "DATA_PAPER":
            row["public_full_text_links"] = ["https://zenodo.org/records/20742676", row["publisher_url"]]
            row["access_grade"] = "PUBLISHER_AND_OFFICIAL_REPOSITORY_OPEN_ACCESS_CONFIRMED"
            row["full_text_access"] = "Published article is open access at publisher and official Zenodo paper record; no experimental files downloaded."
            row["official_dataset_metadata_url"] = "https://zenodo.org/records/20715542"
            row["local_paper_path_relative_to_repository"] = "数据/《耐久性测试期间PEM燃料电池堆的空间解析电化学表征》的基础数据/IJHE_2026_Fensterle_Spatially resolved electrochemical characterization of a PEM fuel cell stack during durability testing.pdf"
            row["local_paper_sha256"] = "c66c0ffa31627637dd2e6842df976731e833bb230b30e0c967d73bcd99fafad1"
            row["content_read_scope"] = "SELECTED_PAGES: PDF p1-3 plus p4 section 2.3 methods text only, stopping before section 3 Results; not full-text read."
            row["source_locators"] = {
                "stack_identity": "p2 section 2.1: durability 10-cell, separate fresh BoL sensitivity 8-cell, 280 cm^2 active area",
                "dynamic_cycle": "p3 section 2.3 and Figure 2: approximately 12 h, three short stops and one longer stop",
                "duration": "p1 abstract says >2000 h; p4 section 2.3 says durability ended after 200 cycles; no EOL claim implied",
                "license": "p1 copyright: CC BY 4.0; independently confirmed by Zenodo metadata.license.id",
            }
        if key == "A3":
            row["metadata_warning"] = "Official Crossref and institutional record list four authors: Ko, Kim, Jaewoong Park, Lee. Some secondary records incorrectly insert Junghwan Park; do not copy that fifth author."
        if key == "P2":
            row["metadata_warning"] = "Preserve publisher/Crossref deposited author names; avoid silently reversing family/given order based on Chinese-name assumptions."
        result.append(row)
    repository = []
    for record_id in ZENODO_IDS:
        path = STAGE / "outputs/evidence_index" / f"ZENODO_{record_id}.json"
        record = json.loads(path.read_text(encoding="utf-8"))
        repository.append({"record_id": record_id, "url": "https://zenodo.org/records/" + record_id,
            "doi": record["metadata"]["doi"], "publication_date": record["metadata"]["publication_date"],
            "title": record["metadata"]["title"], "related_identifiers": record["metadata"].get("related_identifiers", []),
            "license": record["metadata"].get("license"), "access_right": record["metadata"].get("access_right"),
            "version_label_on_record_page": "v1", "machine_version_relations": record["metadata"].get("relations", {}).get("version", []),
            "raw_metadata_path": path.relative_to(STAGE).as_posix(), "raw_metadata_sha256": sha(path),
            "experimental_files_downloaded": False, "metadata_only": True})
    payload = {"search_type": "Targeted nine-paper benchmark plus original TFT and RealHyFC data source; not a systematic review", "checked_date": DATE, "records": result, "official_repository_records": repository}
    target = STAGE / "literature/verified_metadata.json"
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

def verify():
    target = STAGE / "literature/verified_metadata.json"
    payload = json.loads(target.read_text(encoding="utf-8"))
    assert len(payload["records"]) == len(PAPERS)
    for row, (key, doi) in zip(payload["records"], PAPERS):
        assert (row["id"], row["doi"]) == (key, doi)
        path = STAGE / row["raw_crossref_path"]
        assert path.resolve().is_relative_to(STAGE.resolve())
        assert sha(path) == row["raw_crossref_sha256"]
        message = json.loads(path.read_text(encoding="utf-8"))["message"]
        assert message["DOI"].lower() == doi.lower()
        assert row["title"] == message["title"][0]
        assert row["authors"] == [" ".join(filter(None,(a.get("given"),a.get("family")))) or a.get("name", "") for a in message.get("author", [])]
        assert row["journal"] == message.get("container-title", [None])[0]
        assert row["volume"] == message.get("volume")
        assert row["pages"] == message.get("page")
        assert row["article_number"] == message.get("article-number")
        assert row["publisher_issue_date_observed"] == PUBLISHER_ISSUE_DATES[key]
    assert len(payload["official_repository_records"]) == 2
    for row in payload["official_repository_records"]:
        path = STAGE / row["raw_metadata_path"]
        assert path.resolve().is_relative_to(STAGE.resolve())
        assert sha(path) == row["raw_metadata_sha256"]
        message = json.loads(path.read_text(encoding="utf-8"))
        assert str(message["id"]) == row["record_id"]
        assert message["metadata"]["doi"] == row["doi"]
        assert message["metadata"].get("license") == row["license"]
        assert message["metadata"].get("access_right") == row["access_right"]
    print(json.dumps({"status": "PASS", "verified_records": len(PAPERS), "experimental_data_read": False, "mode": "offline official metadata identity and SHA256 verification"}))

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--fetch", action="store_true", help="Network fetch official metadata and rebuild index; default verifies saved evidence offline")
    args = parser.parse_args()
    if args.fetch:
        with ThreadPoolExecutor(max_workers=3) as pool:
            for key, url in pool.map(fetch_one, PAPERS):
                print(json.dumps({"id": key, "source": url, "status": "FETCHED"}))
        for record_id in ZENODO_IDS:
            fetch_zenodo(record_id)
        build()
    verify()
