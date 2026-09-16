# Every environment variable mojoccl reads, in one place.
#
# One constant per variable, named exactly like the variable, so a `getenv`
# call site still reads as the name a user would export and a grep for that
# name finds the definition and every use at once. Nothing else in this
# library spells an environment variable as a literal.
#
# `torch_mojo_backend/env_vars.py` is the project-wide union -- these, the
# base library's (`native/mojo/env_vars.mojo`) and the Python ones -- and is
# what `register_mojo_devices()` checks the user's environment against. The
# three lists build separately and cannot import one another;
# `tests/test_env_vars_are_registered.py` scans the sources and fails if one
# of them grows a name the Python union does not carry.
#
# Semantics live at the `getenv` call site, next to the default and the
# parsing; this file is the index, not a second copy of the documentation.

# --- Transport selection --------------------------------------------------

# `verbs` or `fabric` pins the inter-node transport; unset probes.
comptime MOJOCCL_NET = "MOJOCCL_NET"
# Absolute path of libfabric.so.1, overriding the search.
comptime MOJOCCL_LIBFABRIC = "MOJOCCL_LIBFABRIC"
# Directory for the node-local AF_UNIX bootstrap sockets.
comptime MOJOCCL_SOCKET_DIR = "MOJOCCL_SOCKET_DIR"
# Interface the TCP bootstrap binds to, when the automatic choice is wrong.
comptime MOJOCCL_SOCKET_IFNAME = "MOJOCCL_SOCKET_IFNAME"
# Seconds the TCP bootstrap waits for every rank to check in.
comptime MOJOCCL_BOOTSTRAP_TIMEOUT_S = "MOJOCCL_BOOTSTRAP_TIMEOUT_S"

# --- InfiniBand / verbs ---------------------------------------------------

# Keeps only the named IB device, when a host has several.
comptime MOJOCCL_IB_HCA = "MOJOCCL_IB_HCA"
# `0` drives the queue pairs from the calling thread, not the progress thread.
comptime MOJOCCL_IB_PROXY = "MOJOCCL_IB_PROXY"
# CPU index the progress thread pins to; `none` opts out of pinning.
comptime MOJOCCL_IB_PROXY_CPU = "MOJOCCL_IB_PROXY_CPU"
# Microseconds the progress thread sleeps between idle polls.
comptime MOJOCCL_IB_PROXY_IDLE_US = "MOJOCCL_IB_PROXY_IDLE_US"
# `0` registers memory regions without IBV_ACCESS_RELAXED_ORDERING.
comptime MOJOCCL_IB_RELAXED_ORDERING = "MOJOCCL_IB_RELAXED_ORDERING"
# Seconds a collective waits for its peers before raising the abort word.
comptime MOJOCCL_IB_TIMEOUT_S = "MOJOCCL_IB_TIMEOUT_S"
# `1` prints what each rank negotiated, one line per communicator.
comptime MOJOCCL_IB_TRACE = "MOJOCCL_IB_TRACE"

# --- libfabric ------------------------------------------------------------

# Keeps only the named libfabric domain, so one process per NIC can each
# drive their own.
comptime MOJOCCL_FABRIC_DOMAIN = "MOJOCCL_FABRIC_DOMAIN"
# `0` drops the post-RMA read-back flush.
comptime MOJOCCL_FABRIC_FLUSH = "MOJOCCL_FABRIC_FLUSH"
# `auto` (default), `system`, `rocr` or `cuda`.
comptime MOJOCCL_FABRIC_HMEM = "MOJOCCL_FABRIC_HMEM"
# libfabric provider name; defaults to `cxi`.
comptime MOJOCCL_FABRIC_PROVIDER = "MOJOCCL_FABRIC_PROVIDER"
# Wall-clock -FI_ENOMEM retry budget during endpoint setup, in seconds.
comptime MOJOCCL_FABRIC_SETUP_RETRY_S = "MOJOCCL_FABRIC_SETUP_RETRY_S"

# --- NVLink SHARP (multicast) ---------------------------------------------

# `0` turns the multicast path off, region and all.
comptime MOJOCCL_NVLS = "MOJOCCL_NVLS"
# `rec` sizes the multicast object with the recommended granularity.
comptime MOJOCCL_NVLS_GRANULARITY = "MOJOCCL_NVLS_GRANULARITY"
# MiB at or above which an allreduce takes the multicast path.
comptime MOJOCCL_NVLS_MIN_MB = "MOJOCCL_NVLS_MIN_MB"

# --- Staging memory -------------------------------------------------------

# Size of the registered staging region, in MiB.
comptime MOJOCCL_REGION_MB = "MOJOCCL_REGION_MB"
