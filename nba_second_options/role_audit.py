from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path

import pandas as pd

from .config import PipelineConfig
from .metrics import (
    merge_player_stats,
    normalized_name,
    rank_offensive_roles,
    role_pairing_proposals,
)
from .sources import FrameCache


@dataclass(frozen=True)
class ReviewedPairing:
    primary: str
    secondary: str
    note: str
    status: str = "OWNER_REVIEWED"


# These are not model guesses. They encode corrections explicitly supplied by
# the project owner and remain visible in the audit output.
REVIEWED_PAIRINGS: dict[tuple[str, str], ReviewedPairing] = {
    ("1999-00", "NYK"): ReviewedPairing(
        "Latrell Sprewell", "Allan Houston",
        "Editorial correction: Houston's scoring responsibility puts him ahead of Ewing as the second option.",
        "EDITORIAL_REVIEWED",
    ),
    ("2000-01", "LAL"): ReviewedPairing(
        "Shaquille O'Neal", "Kobe Bryant",
        "Owner-reviewed: Shaq was the primary star; Kobe was the second option.",
    ),
    ("2001-02", "LAL"): ReviewedPairing(
        "Shaquille O'Neal", "Kobe Bryant",
        "Editorial sanity correction: Shaq remained the primary star; Kobe was the second option.",
        "EDITORIAL_REVIEWED",
    ),
    ("2002-03", "DET"): ReviewedPairing(
        "Chauncey Billups", "Richard Hamilton",
        "Owner-reviewed: Billups was the offensive engine; Hamilton was the scoring finisher.",
    ),
    ("2003-04", "DET"): ReviewedPairing(
        "Chauncey Billups", "Richard Hamilton",
        "Owner-reviewed: Billups was the offensive engine; Hamilton was the scoring finisher.",
    ),
    ("2004-05", "DET"): ReviewedPairing(
        "Chauncey Billups", "Richard Hamilton",
        "Owner-reviewed: Billups was the offensive engine; Hamilton was the scoring finisher.",
    ),
    ("2004-05", "PHX"): ReviewedPairing(
        "Steve Nash", "Amar'e Stoudemire",
        "Owner-reviewed: Nash drove the offense; Stoudemire was the lead scoring complement.",
    ),
    ("2005-06", "DET"): ReviewedPairing(
        "Chauncey Billups", "Richard Hamilton",
        "Owner-reviewed: Billups was the offensive engine; Hamilton was the scoring finisher.",
    ),
    ("2006-07", "DET"): ReviewedPairing(
        "Chauncey Billups", "Richard Hamilton",
        "Owner-reviewed: Billups was the offensive engine; Hamilton was the scoring finisher.",
    ),
    ("2006-07", "SAS"): ReviewedPairing(
        "Tim Duncan", "Tony Parker",
        "Owner-reviewed: Duncan was the structural primary; Parker was the second option.",
    ),
    ("2006-07", "UTA"): ReviewedPairing(
        "Deron Williams", "Carlos Boozer",
        "Owner-reviewed: Williams was the offensive engine; Boozer was the scoring complement.",
    ),
    ("2007-08", "SAS"): ReviewedPairing(
        "Tim Duncan", "Tony Parker",
        "Owner-reviewed: Duncan was the structural primary; Parker was the second option.",
    ),
    ("2007-08", "DET"): ReviewedPairing(
        "Chauncey Billups", "Richard Hamilton",
        "Owner-reviewed principle: Billups was the offensive engine; Hamilton was the scoring finisher.",
    ),
    ("2009-10", "BOS"): ReviewedPairing(
        "Paul Pierce", "Kevin Garnett",
        "Owner-reviewed hierarchy; Garnett chosen over Allen as #2 because of his larger usage and two-way responsibility.",
        "EDITORIAL_REVIEWED",
    ),
    ("2009-10", "ORL"): ReviewedPairing(
        "Dwight Howard", "Jameer Nelson",
        "Owner-reviewed: Howard was the structural primary; Nelson was the next scoring/creation option.",
    ),
    ("2009-10", "PHX"): ReviewedPairing(
        "Steve Nash", "Amar'e Stoudemire",
        "Owner-reviewed: Nash remained the offensive engine; Stoudemire was the scoring complement.",
    ),
    ("2010-11", "MIA"): ReviewedPairing(
        "LeBron James", "Dwyane Wade",
        "Owner-reviewed: James was the primary star; Wade was the second option.",
    ),
    ("2010-11", "OKC"): ReviewedPairing(
        "Kevin Durant", "Russell Westbrook",
        "Owner-reviewed: Durant was Oklahoma City's primary star.",
    ),
    ("2011-12", "BOS"): ReviewedPairing(
        "Paul Pierce", "Kevin Garnett",
        "Owner-reviewed: Pierce was the first option; Garnett was the second.",
    ),
    ("2011-12", "OKC"): ReviewedPairing(
        "Kevin Durant", "Russell Westbrook",
        "Owner-reviewed: Durant was Oklahoma City's primary star.",
    ),
    ("2012-13", "MEM"): ReviewedPairing(
        "Marc Gasol", "Zach Randolph",
        "Owner-reviewed: Gasol was the structural centerpiece; Randolph gets #2 on scoring responsibility.",
        "EDITORIAL_REVIEWED",
    ),
    ("2013-14", "OKC"): ReviewedPairing(
        "Kevin Durant", "Russell Westbrook",
        "Owner-reviewed: Durant was Oklahoma City's primary star.",
    ),
    ("2013-14", "SAS"): ReviewedPairing(
        "Tony Parker", "Tim Duncan",
        "Owner-reviewed: Duncan, not Ginobili, was the secondary option.",
    ),
    ("2014-15", "ATL"): ReviewedPairing(
        "Paul Millsap", "Jeff Teague",
        "Owner-reviewed: Millsap was the first option; Teague was the second.",
    ),
    ("2015-16", "OKC"): ReviewedPairing(
        "Kevin Durant", "Russell Westbrook",
        "Owner-reviewed: Durant was Oklahoma City's primary star.",
    ),
    ("2016-17", "GSW"): ReviewedPairing(
        "Kevin Durant", "Stephen Curry",
        "Owner-reviewed: Durant was the primary option; Curry was the second.",
    ),
    ("2017-18", "BOS"): ReviewedPairing(
        "Jayson Tatum", "Jaylen Brown",
        "Editorial correction: committee offense, but Tatum/Brown best represent the playoff scoring hierarchy.",
        "EDITORIAL_REVIEWED",
    ),
    ("2017-18", "GSW"): ReviewedPairing(
        "Kevin Durant", "Stephen Curry",
        "Owner-reviewed: Durant was the primary option; Curry was the second.",
    ),
    ("2018-19", "GSW"): ReviewedPairing(
        "Kevin Durant", "Stephen Curry",
        "Owner-reviewed: Durant was the primary option; Curry was the second.",
    ),
    ("2019-20", "DEN"): ReviewedPairing(
        "Nikola Jokic", "Jamal Murray",
        "Owner-reviewed: Jokic was the offensive engine; Murray was the scoring complement.",
    ),
    ("2019-20", "MIA"): ReviewedPairing(
        "Jimmy Butler III", "Goran Dragic",
        "Owner-reviewed: Butler was Miami's primary star despite Dragic's higher usage.",
    ),
    ("2020-21", "LAC"): ReviewedPairing(
        "Kawhi Leonard", "Paul George",
        "Owner-reviewed: Leonard was the primary star; George was the second option.",
    ),
    ("2021-22", "GSW"): ReviewedPairing(
        "Stephen Curry", "Andrew Wiggins",
        "Research correction: Wiggins combined secondary scoring with the primary Tatum "
        "assignment and finished second on the official Finals MVP ladder.",
        "EDITORIAL_REVIEWED",
    ),
    ("2021-22", "MIA"): ReviewedPairing(
        "Jimmy Butler III", "Bam Adebayo",
        "Owner-reviewed: Adebayo, not Herro, was Miami's second option.",
    ),
    ("2022-23", "DEN"): ReviewedPairing(
        "Nikola Jokic", "Jamal Murray",
        "Owner-reviewed: Jokic was the primary star; Murray was the second option.",
    ),
    ("2023-24", "IND"): ReviewedPairing(
        "Tyrese Haliburton", "Pascal Siakam",
        "Owner-reviewed: Haliburton was the offensive engine; Siakam was the second option.",
    ),
    ("2024-25", "IND"): ReviewedPairing(
        "Tyrese Haliburton", "Pascal Siakam",
        "Owner-reviewed: Haliburton was the offensive engine; Siakam was the second option.",
    ),
    ("2024-25", "NYK"): ReviewedPairing(
        "Jalen Brunson", "Karl-Anthony Towns",
        "Owner-reviewed: Towns was New York's second option.",
    ),
    ("2025-26", "NYK"): ReviewedPairing(
        "Jalen Brunson", "OG Anunoby",
        "Owner-reviewed: Anunoby was New York's second option; Towns is treated as the third star.",
    ),
}


# These are deliberately not overrides. They are obvious cases where the
# numerical proposal can conflict with a structural-star interpretation, so a
# human must settle the row before it can enter the analysis dataset.
SANITY_REVIEW_NOTES: dict[tuple[str, str], str] = {
    ("2000-01", "LAL"): "Resolve Shaq/Kobe ordering; the model narrowly proposes Kobe first.",
    ("2003-04", "IND"): "Resolve Jermaine O'Neal/Artest hierarchy.",
    ("2006-07", "SAS"): "The model proposes Parker first; review Duncan's structural-primary role.",
    ("2007-08", "BOS"): "Potential Garnett/Pierce co-primary structure.",
    ("2007-08", "SAS"): "The model proposes Parker first; review Duncan's structural-primary role.",
    ("2009-10", "ORL"): "The model proposes Nelson first; review Howard's structural-primary role.",
    ("2009-10", "PHX"): "Distinguish Nash's offensive engine role from Stoudemire's scoring lead.",
    ("2010-11", "MIA"): "Potential LeBron/Wade co-primary structure.",
    ("2011-12", "BOS"): "Distinguish Rondo's engine role from Pierce's scoring-option role.",
    ("2015-16", "OKC"): "The model proposes Westbrook first; review Durant's primary-star role.",
    ("2016-17", "GSW"): "Potential Curry/Durant co-primary structure.",
    ("2019-20", "DEN"): "The model proposes Murray first; review Jokic's structural-primary role.",
    ("2021-22", "MIA"): "Resolve Herro/Adebayo as Miami's second option.",
    ("2023-24", "IND"): "Resolve Haliburton's engine role versus Siakam's playoff scoring lead.",
    ("2024-25", "IND"): "Resolve Haliburton's engine role versus Siakam's playoff scoring lead.",
}


def _player_lookup(ranked: pd.DataFrame, team_id: int, player_name: str) -> pd.Series:
    candidates = ranked[
        (ranked["TEAM_ID"] == team_id)
        & (ranked["PLAYER_NAME"].map(normalized_name) == normalized_name(player_name))
    ]
    if candidates.empty:
        raise ValueError(f"Reviewed player {player_name!r} not found for team {team_id}")
    return candidates.iloc[0]


def _set_player_fields(row: dict[str, object], prefix: str, player: pd.Series) -> None:
    row[f"{prefix}_PLAYER_ID"] = int(player["PLAYER_ID"])
    row[prefix] = player["PLAYER_NAME"]
    score_column = "PRIMARY_SCORE" if prefix == "PRIMARY" else "SECONDARY_SCORE"
    row[f"{prefix}_ROLE_SCORE"] = float(player[score_column])
    row[f"{prefix}_PPG"] = float(player["PPG"])
    row[f"{prefix}_USG_PCT"] = float(player["USG_PCT"])
    row[f"{prefix}_AST_PCT"] = float(player["AST_PCT"])
    row[f"{prefix}_SCORING_LOAD"] = float(player["SCORING_LOAD"])
    row[f"{prefix}_CREATION_ENGINE"] = float(player["CREATION_ENGINE"])
    row[f"{prefix}_IMPACT_SIGNAL"] = float(player["IMPACT_SIGNAL"])


def build_role_audit(
    cache_dir: Path,
    runs_csv: Path,
    config: PipelineConfig,
) -> pd.DataFrame:
    """Create proposals for every Conference Finals team without selecting runs.

    The existing run file is used only to identify the four qualifying teams in
    each season. Its old usage-based player selections are deliberately ignored.
    """
    runs = pd.read_csv(runs_csv)
    teams = runs[["SEASON", "TEAM_ID", "TEAM_ABBREVIATION"]].drop_duplicates()
    cache = FrameCache(cache_dir)
    audit_frames: list[pd.DataFrame] = []

    for season, season_teams in teams.groupby("SEASON", sort=True):
        base = cache.read("nba_stats", f"players_{season}_Base_all")
        advanced = cache.read("nba_stats", f"players_{season}_Advanced_all")
        if base is None or advanced is None:
            raise FileNotFoundError(f"Missing cached player tables for {season}")

        players = merge_player_stats(base, advanced)
        players = players[players["TEAM_ID"].isin(season_teams["TEAM_ID"])].copy()
        ranked = rank_offensive_roles(players, config)
        proposals = role_pairing_proposals(ranked)
        proposals["SEASON"] = season
        proposals["MODEL_PRIMARY"] = proposals["PRIMARY"]
        proposals["MODEL_SECONDARY"] = proposals["SECONDARY"]
        proposals["PAIRING_SOURCE"] = "MODEL_PROPOSAL"
        proposals["EDITORIAL_NOTE"] = ""

        for index, proposal in proposals.iterrows():
            key = (season, str(proposal["TEAM_ABBREVIATION"]))
            reviewed = REVIEWED_PAIRINGS.get(key)
            if reviewed is None:
                sanity_note = SANITY_REVIEW_NOTES.get(key)
                if sanity_note:
                    proposals.loc[index, "ROLE_REVIEW_STATUS"] = "EDITORIAL_REVIEW_REQUIRED"
                    proposals.loc[index, "EDITORIAL_NOTE"] = sanity_note
                continue
            row = proposal.to_dict()
            primary = _player_lookup(ranked, int(proposal["TEAM_ID"]), reviewed.primary)
            secondary = _player_lookup(ranked, int(proposal["TEAM_ID"]), reviewed.secondary)
            _set_player_fields(row, "PRIMARY", primary)
            _set_player_fields(row, "SECONDARY", secondary)
            remaining = ranked[
                (ranked["TEAM_ID"] == int(proposal["TEAM_ID"]))
                & ~ranked["PLAYER_ID"].isin(
                    [int(primary["PLAYER_ID"]), int(secondary["PLAYER_ID"])]
                )
            ].sort_values("SECONDARY_SCORE", ascending=False)
            challenger = remaining.iloc[0] if not remaining.empty else None
            row["THIRD"] = challenger["PLAYER_NAME"] if challenger is not None else ""
            row["THIRD_ROLE_SCORE"] = (
                float(challenger["SECONDARY_SCORE"])
                if challenger is not None
                else float("nan")
            )
            row["HIERARCHY_MARGIN"] = float(
                primary["ROLE_SCORE"] - secondary["ROLE_SCORE"]
            )
            row["SECONDARY_MARGIN"] = (
                float(secondary["SECONDARY_SCORE"] - challenger["SECONDARY_SCORE"])
                if challenger is not None
                else float("nan")
            )
            row["ROLE_REVIEW_STATUS"] = reviewed.status
            row["PAIRING_SOURCE"] = reviewed.status
            row["EDITORIAL_NOTE"] = reviewed.note
            proposals.loc[index, list(row)] = list(row.values())

        audit_frames.append(proposals)

    audit = pd.concat(audit_frames, ignore_index=True)
    audit["FINAL_PRIMARY"] = audit["PRIMARY"]
    audit["FINAL_SECONDARY"] = audit["SECONDARY"]
    return audit.sort_values(["SEASON", "TEAM_ABBREVIATION"]).reset_index(drop=True)


def audit_markdown(audit: pd.DataFrame) -> str:
    lines = [
        "# #1 / #2 Role Audit — Conference Finalists Since 2000",
        "",
        "This file shows every #1/#2 decision before the ranking is calculated. The pairing should match how the team actually played in that postseason, not simply who ranked first and second in one stat.",
        "",
        "The first pass scores the primary star using scoring load (40%), creation (30%), PIE (25%), and minutes (5%). Once that player is removed, the second-option pass uses scoring load (65%), creation (10%), PIE (20%), and minutes (5%). Close calls are flagged. `OWNER_REVIEWED` means I changed or confirmed the pairing after a basketball review; the other rows remain model proposals.",
        "",
        "Lineup net rating is intentionally excluded from role identification.",
        "",
    ]
    for season, group in audit.groupby("SEASON", sort=True):
        lines.extend([f"## {season}", ""])
        for _, row in group.iterrows():
            evidence = (
                f"#1 {row.PRIMARY_PPG:.1f} PPG / {100 * row.PRIMARY_USG_PCT:.1f}% USG; "
                f"#2 {row.SECONDARY_PPG:.1f} PPG / {100 * row.SECONDARY_USG_PCT:.1f}% USG; "
                f"engine indices {row.PRIMARY_CREATION_ENGINE:.0f}/{row.SECONDARY_CREATION_ENGINE:.0f}"
            )
            challenger = f"; #3 challenger: {row.THIRD}" if row.THIRD else ""
            note = f" — {row.EDITORIAL_NOTE}" if row.EDITORIAL_NOTE else ""
            lines.append(
                f"- [ ] **{row.TEAM_ABBREVIATION}: {row.PRIMARY} → {row.SECONDARY}** "
                f"(`{row.ROLE_REVIEW_STATUS}`; {evidence}{challenger}){note}"
            )
        lines.append("")
    lines.extend([
        "## Review rule",
        "",
        "Approve a row only if #1 represents the postseason's structural primary star and #2 represents the next offensive option. If the team had a genuinely fluid hierarchy, mark it `CO_PRIMARY` or `COMMITTEE` rather than forcing false precision.",
    ])
    return "\n".join(lines)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build the #1/#2 editorial role audit.")
    parser.add_argument("--cache-dir", type=Path, default=Path("../work/data_cache"))
    parser.add_argument(
        "--runs-csv", type=Path, default=Path("analysis/second_option_runs.csv")
    )
    parser.add_argument("--output-dir", type=Path, default=Path("analysis"))
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    audit = build_role_audit(
        args.cache_dir,
        args.runs_csv,
        PipelineConfig(cache_dir=args.cache_dir, output_dir=args.output_dir, offline=True),
    )
    audit.to_csv(args.output_dir / "role_pairing_audit.csv", index=False)
    (args.output_dir / "role_pairing_audit.md").write_text(audit_markdown(audit))
    print(f"Wrote {len(audit)} role proposals across {audit.SEASON.nunique()} seasons")


if __name__ == "__main__":
    main()
