# mojoccl transport self-tests that need no GPU

Two standalone Mojo programs that exercise
`torch_mojo_backend/distributed/mojoccl/{bootstrap,ibverbs,internode}.mojo`
on any host with InfiniBand — the SLURM **login node** included, which is
what makes them cheap enough to run on every change. They caught six real
bugs (bootstrap/QP/immediate wiring, resource leaks on a failed `ib_setup`,
a silently-misread port LID) before any GPU time was spent chasing them.

Build (from a checkout of this repo, no accelerator needed):

    uv run --no-sync mojo build tests/multinode/selftest/bs_test.mojo \
        -I torch_mojo_backend/distributed/mojoccl -o /tmp/bs_test
    uv run --no-sync mojo build tests/multinode/selftest/ib_bringup.mojo \
        -I torch_mojo_backend/distributed/mojoccl -o /tmp/ib_bringup

## `bs_test.mojo` — the TCP bootstrap

`bs_test <rank> <nranks> <uid-file> <fake-host-index>`. Rank 0 writes the
128-byte unique id to `<uid-file>`; the rest poll for it. Each rank adds
`<fake-host-index>` to its host hash, so one box can pretend to be several
nodes and the derived node/local_rank table can be checked against a known
answer. Runs both blob rounds (16-byte and 256-byte, payload verified) and
the closing barrier.

    for r in $(seq 0 15); do /tmp/bs_test $r 16 /tmp/uid.txt $((r/8)) & done; wait

Exercised: 8 ranks/1 node, 16/2, 6/3.

## `ib_bringup.mojo` — the RDMA transport

`ib_bringup <rank> <nranks> <uid-file>`. Every rank is its own "node"
(local_world 1), so `nranks-1` queue pairs are created per rank. Registers
host memory as the region, runs the bootstrap, moves every QP to RTS, then
runs 400 exchanges through `internode.ib_exchange_now` — the inline
equivalent of the stream callback — alternating inbox halves and verifying
every peer's slot against a rank/sequence-derived pattern.

    for r in 0 1 2 3; do
        MOJOCCL_IB_PROXY=0 /tmp/ib_bringup $r 4 /tmp/uid4.txt &
    done; wait

`MOJOCCL_IB_PROXY=0` is required here and only here: the progress thread's
mailbox is pinned, device-mapped host memory, which needs a driver that can
allocate it, and this host has no GPU. Without it `ib_setup` fails with
"symbol not found: cuMemHostAlloc" -- which is the honest answer, not
something to paper over, since on a real node that symbol is always there.

400 exchanges is deliberately past `RECV_DEPTH` (64): a receive WR that is
consumed and not reposted shows up as a hang rather than as wrong data.

Covered by these and NOT by anything that needs a GPU: interface selection,
the unique-id encoding, the two-round rendezvous, topology derivation, HCA
and port selection, `ibv_reg_mr`, QP INIT/RTR/RTS with NCCL's attribute
values, the ops-table dispatch for post_send/post_recv/poll_cq, the
immediate's parity tagging, recv reposting, the self-QP GPUDirect flush, and
`ib_setup`'s unwind of partially-created resources on a failure path. NOT
covered: registration of *device* memory (needs nvidia_peermem and a GPU),
the progress thread and its two spin kernels (they need pinned host memory
and a stream), and everything in `mojoccl.mojo` above the transport.
