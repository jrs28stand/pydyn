# This file is part of pydyn. GPL-3.0-or-later (see LICENSE).
"""File functions of .mod files inside pydyn.dynare(): isfile/isfolder/copyfile/mkdir/delete work inside the .mod file's folder
only, and never overwrite or delete a file pydyn did not write (the sandbox rule of ``save``)."""
from pathlib import Path

import pytest

import pydyn
from pydyn import ResultsError

REF = Path(__file__).resolve().parent.parent / "reference" / "oo"


def _mod(tmp_path, body, name="f.mod"):
    src = (REF / "ep_back.mod").read_text().replace("extended_path(order=0, periods=40);", "extended_path(order=0, periods=5);\n" + body)
    (tmp_path / name).write_text(src)
    return tmp_path / name


def test_mkdir_copyfile_isfile_delete_round_trip(tmp_path):
    body = ("Simulated_time_series.save('mine');\n"
            "mkdir('sub/deeper');\n"
            "[st, msg] = mkdir('sub/deeper');\n"
            "copyfile('mine.mat', 'sub/deeper/copy.mat');\n"
            "copyfile('mine.mat', 'sub');\n"
            "a = isfile('sub/deeper/copy.mat') && isfile('sub/mine.mat') && isfolder('sub/deeper') && ~isfile('sub');\n"
            "b = exist('sub/deeper', 'dir');\n"
            "delete('sub/deeper/copy.mat');\n"
            "c = ~isfile('sub/deeper/copy.mat');\n")
    ws = pydyn.dynare(_mod(tmp_path, body))
    assert bool(ws.vars["a"]) and bool(ws.vars["c"]) and float(ws.vars["b"]) == 7 and float(ws.vars["st"]) == 1
    assert ws.vars["msg"] == "directory exists"
    assert (tmp_path / "sub" / "deeper").is_dir() and (tmp_path / "sub" / "mine.mat").is_file()
    assert not (tmp_path / "sub" / "deeper" / "copy.mat").exists()


def test_copyfile_never_overwrites_foreign_files(tmp_path):
    (tmp_path / "keep.mat").write_bytes(b"not ours")
    with pytest.raises(ResultsError, match="was not written by pydyn"):
        pydyn.dynare(_mod(tmp_path, "Simulated_time_series.save('mine');\ncopyfile('mine.mat', 'keep.mat');\n"))
    assert (tmp_path / "keep.mat").read_bytes() == b"not ours"
    (tmp_path / "sub").mkdir()
    (tmp_path / "sub" / "foreign.mat").write_bytes(b"not ours either")
    with pytest.raises(ResultsError, match="was not written by pydyn"):
        pydyn.dynare(_mod(tmp_path, "delete('sub/foreign.mat');\n", "g.mod"))
    assert (tmp_path / "sub" / "foreign.mat").exists()


@pytest.mark.parametrize("stmt", ["copyfile('../x.mat', 'a.mat');", "copyfile('f.mod', '../y.mat');", "mkdir('../other');", "mkdir('/tmp/zz_pydyn');",
                                  "r = isfile('../f.mod');", "r = isfolder('/tmp');", "delete('../f.mod');", "system('ls');"])
def test_paths_outside_the_model_folder_and_shell_functions_are_refused(tmp_path, stmt):
    with pytest.raises(ResultsError):
        pydyn.dynare(_mod(tmp_path, stmt + "\n"))
    assert not (tmp_path.parent / "other").exists() and not (tmp_path.parent / "y.mat").exists()


def test_refusal_names_the_function(tmp_path):
    with pytest.raises(ResultsError, match=r"fopen: file-system, shell or interactive functions are not available"):
        pydyn.dynare(_mod(tmp_path, "fid = fopen('x.txt', 'w');\n"))
