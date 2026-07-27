from controller.external_services.cache_controller import CacheController
import sys


class CacheCLI:
    def __init__(self, conf):
        self.conf = conf

    def handle_cache_command(
        self,
        cache_action: str,
        force: bool = False,
        res_url: str = None,
        output_path: str = None,
        filter_pattern: str = None,
        file_path: str = None,
    ):
        """Handle cache subcommands"""
        cache_mgr = CacheController(self.conf)

        if cache_action == "clear":
            cache_mgr.clear_cache(force=force)

        elif cache_action == "get":
            cache_mgr.get_resource(res_url, output_path)

        elif cache_action == "delete":
            cache_mgr.delete_resource(res_url)

        elif cache_action == "list":
            cache_mgr.list_resources(filter_pattern)

        elif cache_action == "stats":
            cache_mgr.show_stats()

        elif cache_action == "add":
            cache_mgr.add_resource(file_path)
        else:
            print(
                "Error: No cache action specified. Use --help for available commands."
            )
            sys.exit(1)
