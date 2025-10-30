"""
the drugs_list.yml file can be updated by the user and stored in version control

this file represents a list of illnesses and the medications that are commonly prescribed

the illnesses can each have tags associating them with groups, forexample the loneliness tag that indicates an illness,
and therefore all associated medications for that illness are associated with loneliness
"""

from pathlib import Path
from typing import Annotated, Never, overload
from pydantic import AfterValidator, BaseModel, RootModel, validate_call
from pydantic_yaml import parse_yaml_file_as, to_yaml_file

from loneliness_prescription_index.config import DATA_DIR

TitleCaseStr = Annotated[str, AfterValidator(str.title)]


class Medication(BaseModel, frozen=True):
    name: TitleCaseStr


class Illness(BaseModel, frozen=True):
    illness: TitleCaseStr
    tags: list[TitleCaseStr]
    medications: list[Medication]


class IllnessConfig(RootModel[dict[TitleCaseStr, Illness]], frozen=True):
    """validation for the medication list.

    Will ensure all strings are 'Title Case' thereby matching the 'chemical_substance_bnf_descr' field in the prescriptions api

    loads and validates the drugs_list.yml file, or alternatively a different user specified file provided"""

    @classmethod
    def load(cls, path: str | Path = DATA_DIR / "drugs_list.yml"):
        return parse_yaml_file_as(IllnessConfig, path)

    def save(self, path: str | Path = DATA_DIR / "drugs_list.yml"):
        return to_yaml_file(path, self)

    @property
    def all_tags(self):
        return {tag for illness in self.root.values() for tag in illness.tags}

    @property
    def all_illnesses(self):
        return {x for x in self.root}

    @overload
    def get_medications(self, *, tag: TitleCaseStr, illness: TitleCaseStr) -> Never: ...

    @overload
    def get_medications(
        self, *, tag: TitleCaseStr, illness: None = None
    ) -> set[Medication]: ...

    @overload
    def get_medications(
        self, *, tag: None = None, illness: TitleCaseStr
    ) -> set[Medication]: ...

    @overload
    def get_medications(
        self, *, tag: None = None, illness: None = None
    ) -> set[Medication]: ...

    @validate_call
    def get_medications(
        self, tag: TitleCaseStr | None = None, illness: TitleCaseStr | None = None
    ) -> set[Medication]:
        """gets the medications in the yaml file.

        optonally the user can provide either a tag or an illness to use as a filter to get all the medications associated with that value

        values for tag and illenss are transformed to title case automatically

        Medications are used to extract a name or file name associated with a group"""
        match (tag, illness):
            case (None, None):
                med_set = {
                    medication
                    for item in self.root.values()
                    for medication in item.medications
                }
            case (t, None):
                med_set = {
                    medication
                    for item in self.root.values()
                    for medication in item.medications
                    if t in item.tags
                }
            case (None, i):
                med_set = set(self.root[i].medications)
            case (t, i):
                raise ValueError("cannot set tag and illness at the same time")
        return med_set


if __name__ == "__main__":
    # integration tests to validate the file exists, is formatted correctly, and can extract medications based on illness or tag
    illness_list = IllnessConfig.load()
    print(illness_list)
    print(illness_list.get_medications(illness="social anxiety"))
    print(illness_list.get_medications(tag="Loneliness"))
