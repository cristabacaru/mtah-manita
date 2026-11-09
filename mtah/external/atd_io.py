"""Read / write tracking in the competition's ATD format (FIFA EPTS: .txt + .xml metadata).

Writing a validation game as a real ATD pair means the *same* parser and pipeline that run on the
competition match (starter/baseline.py, P1's parser, kloppy metrica_epts) run on it unchanged.

Text line:  frame_id:x,y,s;x,y,s;...(S slots):bx,by,bz      x, y as pitch fractions, NaN if unseen
The XML follows the layout of the public Metrica Sample_Game_3 metadata (same provider, same
format): Team (home first), Player per slot with teamId, PlayerChannels and one
DataFormatSpecification. frame_id = row + 1.
"""

import os
import sys
from xml.sax.saxutils import escape

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(HERE))
sys.path.insert(0, ROOT)

from mtah.schema import Game, HOME, write_gt_csv  # noqa: E402


def _fmt(a, digits):
    """Format a float array as strings, NaN -> 'NaN'."""
    s = np.char.mod("%.{}f".format(digits), np.nan_to_num(a, nan=0.0))
    return np.where(np.isfinite(a), s, "NaN")


def write_txt(game, path):
    L, W = game.pitch
    n, S = game.n, game.players.shape[1]
    px = _fmt(game.players[..., 0] / L, 5)
    py = _fmt(game.players[..., 1] / W, 5)
    ps = _fmt(game.players[..., 2], 2)
    bx = _fmt(game.ball[:, 0] / L, 5)
    by = _fmt(game.ball[:, 1] / W, 5)
    bz = _fmt(game.ball[:, 2], 2)
    chunk = 20000
    with open(path, "w") as fh:
        for a in range(0, n, chunk):
            b = min(n, a + chunk)
            groups = np.char.add(np.char.add(np.char.add(np.char.add(px[a:b], ","), py[a:b]), ","),
                                 ps[a:b])
            lines = []
            for i in range(b - a):
                lines.append("{}:{}:{},{},{}\n".format(a + i + 1, ";".join(groups[i]),
                                                        bx[a + i], by[a + i], bz[a + i]))
            fh.writelines(lines)
    return path


def write_xml(game, path, home_name="Home", away_name="Away"):
    L, W = game.pitch
    n, S = game.n, game.players.shape[1]
    dirs = ("left_to_right", "right_to_left")
    home_dir = dirs[0] if game.home_attacks_right_first_half else dirs[1]
    away_dir = dirs[1] if game.home_attacks_right_first_half else dirs[0]
    params = []
    names = ("first_half", "second_half", "first_extra_half", "second_extra_half")
    for (a, b), nm in zip(game.halves, names):
        params += [(nm + "_start", a + 1), (nm + "_end", b)]
    out = ['<?xml version="1.0" encoding="UTF-8"?>',
           '<main xmlns:xsi="http://www.w3.org/2001/XMLSchema-instance" '
           'xsi:schemaLocation="FIFADataTransferFormatEPTSNamespace/FIFADataTransferFormatEPTS '
           'FIFADataTransferFormatEPTS.xsd">',
           "<Metadata>", "<GlobalConfig>",
           "<FileName>{}</FileName>".format(escape(game.name)),
           "<TrackingType>Optical</TrackingType>",
           "<ProviderName>mtah-simulated</ProviderName>",
           "<FrameRate>{}</FrameRate>".format(game.fps), "<ProviderGlobalParameters>"]
    out += ["<ProviderParameter><Name>{}</Name><Value>{}</Value></ProviderParameter>".format(k, v)
            for k, v in params]
    out += ["</ProviderGlobalParameters>", "</GlobalConfig>",
            '<Sessions><Session id="session1"><SessionType>Match</SessionType>'
            '<MatchParameters><Score idLocalTeam="TH" idVisitingTeam="TA">'
            "<LocalTeamScore>0</LocalTeamScore><VisitingTeamScore>0</VisitingTeamScore>"
            "</Score></MatchParameters>"
            "<FieldSize><Width>{:g}</Width><Height>{:g}</Height></FieldSize>".format(L, W),
            "</Session></Sessions>", "<Teams>"]
    for tid, name, d in (("TH", home_name, home_dir), ("TA", away_name, away_dir)):
        out.append('<Team id="{}"><Name>{}</Name><ProviderTeamsParameters><ProviderParameter>'
                   "<Name>attack_direction_first_half</Name><Value>{}</Value>"
                   "</ProviderParameter></ProviderTeamsParameters></Team>".format(tid, escape(name), d))
    out += ["</Teams>", "<Players>"]
    for s in range(S):
        tid = "TH" if game.team[s] == HOME else "TA"
        out.append('<Player id="P{0}" teamId="{1}"><Name>Track {0}</Name>'
                   "<ShirtNumber>{0}</ShirtNumber></Player>".format(s + 1, tid))
    out += ["</Players>", '<Devices><Device id="device1"><Name>Device 1</Name><Sensors>'
            '<Sensor id="position"><Name>Position</Name><Channels>'
            '<Channel id="x"><Name>position_x</Name><Unit>normalized</Unit></Channel>'
            '<Channel id="y"><Name>position_y</Name><Unit>normalized</Unit></Channel>'
            '<Channel id="z"><Name>position_z</Name><Unit>meters</Unit></Channel>'
            '<Channel id="speed"><Name>speed</Name><Unit>meters per second</Unit></Channel>'
            "</Channels></Sensor></Sensors></Device></Devices>", "<PlayerChannels>"]
    for s in range(S):
        for c in ("x", "y", "speed"):
            out.append('<PlayerChannel id="player{0}_{1}" playerId="P{0}" channelId="{1}"/>'
                       .format(s + 1, c))
    out += ["</PlayerChannels>", "</Metadata>", "<DataFormatSpecifications>",
            '<DataFormatSpecification startFrame="1" endFrame="{}" separator=":">'.format(n),
            '<StringRegister name="frameCount"/>', '<SplitRegister separator=";">']
    for s in range(S):
        out.append('<SplitRegister separator=",">' + "".join(
            '<PlayerChannelRef playerChannelId="player{}_{}"/>'.format(s + 1, c)
            for c in ("x", "y", "speed")) + "</SplitRegister>")
    out += ["</SplitRegister>", '<SplitRegister separator=",">'
            '<BallChannelRef channelId="x"/><BallChannelRef channelId="y"/>'
            '<BallChannelRef channelId="z"/></SplitRegister>',
            "</DataFormatSpecification>", "</DataFormatSpecifications>", "</main>"]
    with open(path, "w") as fh:
        fh.write("\n".join(out) + "\n")
    return path


def write(game, folder, stem=None):
    """Write <stem>_atd.txt, <stem>_atd.xml and (if labelled) <stem>_gt.csv into folder."""
    stem = stem or game.name
    os.makedirs(folder, exist_ok=True)
    base = os.path.join(folder, stem)
    paths = {"atd_raw": write_txt(game, base + "_atd.txt"),
             "atd_meta": write_xml(game, base + "_atd.xml",
                                   game.meta.get("home", "Home"), game.meta.get("away", "Away"))}
    if game.labels is not None:
        paths["gt"] = write_gt_csv(base + "_gt.csv", game.labels, game.fps, game.halves)
    return paths


def read_txt(path, n_slots):
    """(frame_id (n,), players (n, S, 3) fractions+speed, ball (n, 3) fractions+z).

    Vectorised: every line has the same number of fields, so the whole file is one float array.
    """
    width = 1 + 3 * n_slots + 3
    rows = []
    with open(path) as fh:
        while True:
            lines = fh.readlines(64 << 20)          # ~64 MB at a time keeps memory flat
            if not lines:
                break
            text = "".join(lines).replace(":", ",").replace(";", ",").replace("\n", ",")
            vals = np.fromstring(text, sep=",")      # parses NaN; text mode is not deprecated
            if vals.size % width:
                raise ValueError("{}: lines do not all have {} fields".format(path, width))
            rows.append(vals.reshape(-1, width))
    values = np.concatenate(rows)
    frame = values[:, 0].astype(np.int64)
    players = values[:, 1:1 + 3 * n_slots].reshape(-1, n_slots, 3)
    ball = values[:, 1 + 3 * n_slots:]
    return frame, players, ball


def team_of_slots(xml_path):
    """Slot -> 0 home / 1 away, home = first Team in the file (same rule as starter/baseline)."""
    from lxml import etree
    root = etree.parse(xml_path).getroot()
    teams = [t.get("id") for t in root.iter("Team")]
    return np.array([0 if p.get("teamId") == teams[0] else 1 for p in root.iter("Player")],
                    np.int8)
