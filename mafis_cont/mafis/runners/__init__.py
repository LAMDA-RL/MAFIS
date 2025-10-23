"""Runner registry."""
from mafis.runners.off_policy_ha_runner import OffPolicyHARunner

RUNNER_REGISTRY = {
    "hasac": OffPolicyHARunner,
}
