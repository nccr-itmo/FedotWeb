from init.init_cases import create_default_cases
from init.init_history import create_default_history
from init.init_pipelines import create_default_pipelines


if __name__ == "__main__":
    create_default_cases()
    create_default_pipelines()
    create_default_history(opt_times=[1, 0.25, 0.25])
