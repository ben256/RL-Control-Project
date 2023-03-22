from environments.fixed_mass_environment import FixedMassEnvironment


def select_env(env_name):
    if env_name == "FixedMassEnvironment":
        return FixedMassEnvironment()
    else:
        raise ValueError("Unknown environment name")