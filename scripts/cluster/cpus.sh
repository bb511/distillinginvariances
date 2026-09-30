#!/usr/bin/env bash
# Print the CPU ids this pod may use, as the kernel lists them (e.g. 3-14,51-62), for
# taskset -c. A pod's cpuset is neither 0-based nor contiguous, so 0-$(nproc) would pin
# workers to cores the pod does not own. Run inside a pod.
set -euo pipefail

# cgroup v2 exposes the effective set directly; under v1 the process's own mask carries it.
if [ -r /sys/fs/cgroup/cpuset.cpus.effective ]; then
    cat /sys/fs/cgroup/cpuset.cpus.effective
else
    awk '/^Cpus_allowed_list:/ {print $2}' /proc/self/status
fi
