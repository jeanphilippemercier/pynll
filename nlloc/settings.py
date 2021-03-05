from dynaconf import LazySettings
import os


class Settings(LazySettings):
    def __init__(self):
        """
        Init function currently just initializes the object allowing
        """

        if "NLLOC_CONFIG" in os.environ:
            config_dir = os.environ['NLLOC_CONFIG']
        else:
            config_dir = os.getcwd()

        if "NLL_ROOT" in os.environ:
            # keep this as legacy behavior
            nlloc_root_dir = os.environ['NLLOC_ROOT']
        else:
            nlloc_root_dir = os.getcwd()

        dconf = {}
        dconf.setdefault('ENVVAR_PREFIX_FOR_DYNACONF', 'NLLOC')

        env_prefix = '{0}_ENV'.format(
            dconf['ENVVAR_PREFIX_FOR_DYNACONF']
        )  # NLL_ENV

        dconf.setdefault(
            'ENV_FOR_DYNACONF',
            os.environ.get(env_prefix, 'DEVELOPMENT').upper()
        )

        settings_path = os.path.join(os.path.dirname(os.path.realpath(
                           __file__)), "settings.toml")

        default_paths = (f'{settings_path},'
                         f'settings.py,.secrets.py,settings.toml,settings.tml,'
                         f'.secrets.toml,.secrets.tml,settings.yaml,'
                         f'settings.yml,.secrets.yaml,.secrets.yml,'
                         f'settings.ini,settings.conf,settings.properties,'
                         f'connectors.toml,connectors.tml,.connectors.toml,'
                         f'.connectors.tml,connectors.json,.secrets.ini,'
                         f'.secrets.conf,.secrets.properties,settings.json,'
                         f'.secrets.json')

        dconf['SETTINGS_FILE_FOR_DYNACONF'] = default_paths
        dconf['ROOT_PATH_FOR_DYNACONF'] = config_dir
        dconf['NLLOC_ROOT_DIR'] = nlloc_root_dir

        super().__init__(**dconf)


settings = Settings()