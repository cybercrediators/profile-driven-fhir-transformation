from data_handling.registry.registry_object import RegistryObject

from typing import List, Dict


class Registry:
    """Local registry for objects of the given FHIR profile."""

    def __init__(self):
        self.registry_objects: Dict[str, RegistryObject] = {}
        self.examples = {}
        self.working_objects = {}

    def add_fhir_object(self, fhir_obj: object, res_type: str = None) -> RegistryObject:
        """Add a FHIR object to the local registry"""
        # print(fhir_obj.url)
        if not hasattr(fhir_obj, "url"):
            raise ReferenceError("Object does not have a canonical URL!")
        if self.registry_objects.get(fhir_obj.url):
            raise ReferenceError("Object already stored in the registry!")
        self.registry_objects[fhir_obj.url] = RegistryObject(fhir_obj, res_type)
        return self.registry_objects[fhir_obj.url]

    # def replace_fhir_object(self, old, new):
    #     pass

    def get_all_mappable_fields(self) -> List[Dict[str, List[str]]]:
        """Return the mappable fields of all objects in the registry, keyed by URL."""
        return [
            {url: obj.mappable_fields} for url, obj in self.registry_objects.items()
        ]

    def get_all_mappable_fields_of_roots(self) -> List[Dict[str, List[str]]]:
        """Return the mappable fields of all root objects in the registry, keyed by URL."""
        return [
            {url: obj.mappable_fields}
            for url, obj in self.registry_objects.items()
            if obj.is_root
        ]

    def get_all_unprocessed_objects(self) -> List[str]:
        """Return the URLs of all objects in the registry that are not yet processed."""
        return [url for url, obj in self.registry_objects.items() if obj.processed == 0]

    def get_obj_by_name(self, obj_name: str) -> RegistryObject:
        """Get a registry object by its canonical URL, or None if absent."""
        return self.registry_objects.get(obj_name)

    def get_all_obj_by_type(self, res_type: str) -> List[RegistryObject]:
        """Get all registry objects of the given resource type."""
        obj_names = self.get_all_obj_names_by_type(res_type)
        return [self.get_obj_by_name(obj) for obj in obj_names]

    def get_all_obj_names_by_type(self, res_type: str) -> List[str]:
        """Get the URLs of all registry objects of the given resource type."""
        return list(
            filter(
                lambda k: self.registry_objects[k].res_type == res_type,
                self.registry_objects,
            )
        )

    def add_to_used_by(self, src_url: str, target_url: str) -> None:
        """
        Add the target_url to the object of the src_url.used_by set
        (if inside registry)
        i.e.: src_url obj is used by target_url obj
        OR in other words: target_url obj uses the src_url obj

        :param self: Description
        :param src_url: Description
        :param target_url: Description
        """
        if src_obj := self.get_obj_by_name(src_url):
            src_obj.used_by.add(target_url)
