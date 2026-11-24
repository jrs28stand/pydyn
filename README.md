# Large Octave reference results (Dynare 7.2 test suite), excluding heterogeneous-agent models

This is a companion to branch `accept-ref-octave`, whose README gives the provenance, the run conditions and `index.csv`: status, elapsed time and `mat_sha256` for every test. This branch holds the 16 result files over 5 MB that are left out of that branch's archive, except the heterogeneous-agent (HANK and Krusell–Smith) models.

The archive is in nine parts, `accept_ref_large.tar.gz.part00`–`part08`, 393 MB in total. To reassemble and unpack it:

    cat accept_ref_large.tar.gz.part* > accept_ref_large.tar.gz
    sha256sum accept_ref_large.tar.gz   # 35c6ba8666eee992698264850b178a0d455da49eaf2d98c73491d85969ba6f8f
    tar xzf accept_ref_large.tar.gz     # adds files under accept_ref/

Unpack it in the same directory as the `accept-ref-octave` archive: the files land in place in the same `accept_ref/` tree. `files.txt` lists them. Each file's sha256 is in `index.csv` on `accept-ref-octave`.

The 10 heterogeneous-agent result files (284 MB) are not published, because neither port supports these models. Regenerate them with `tools/accept_octave.py --only heterogeneity` if they are needed.
