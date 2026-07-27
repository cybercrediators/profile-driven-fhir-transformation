"""
Create custom FHIR processors for Nifi based on the FHIR shorthand format
"""

import logging

from view import options
from controller.service_controller import ServiceController

if "__main__" == __name__:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    args = options.get_args()

    service_controller = ServiceController(conf_path=args.config)
    service_controller.handle_command(args)

    ## Handle processing (default behavior)
    ## Init pipeline controller if in development mode
    # elif args.development:
    #     pipeline_controller = PipelineController(args)
    #     pipeline_controller.initial_processing()
    #     print(f"Setup validation: {pipeline_controller.validate_setup()}")
    # else:
    #     pipeline_controller = PipelineController(
    #         conf_path=getattr(args, 'config', "conf/default.json"),
    #         overwrite_all=getattr(args, 'overwrite_all', False),
    #         reprocess_profile=getattr(args, 'reprocess_profile', False),
    #         reprocess_helper_definition=getattr(args, 'reprocess_helper_definition', False),
    #         reprocess_structure_map=getattr(args, 'reprocess_structure_map', False),
    #         create_references=getattr(args, 'create_references_in_structure_map', False),
    #         minimal_mode=getattr(args, 'minimal_structure_map', False),
    #         automapping=getattr(args, 'auto_mapping', False),
    #         tarred_profile=getattr(args, 'tarred_profile', ""),
    #     )
    #     pipeline_controller.initial_processing()
    #     if args.prepare_matchbox:
    #         pipeline_controller.prepare_matchbox_setup()
    #     if args.show_config:
    #         pipeline_controller.display_config()
    #         sys.exit(0)
    #     sys.exit(0)
