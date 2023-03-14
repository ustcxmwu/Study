#  Copyright (c) 2021. Xiaomin Wu <xmwu@mail.ustc.edu.cn>
#  All rights reserved.
import typing as t

import yaml
from validx import Float, Validator, contracts
from validx.exc import ValidationError, OptionsError


class InitFloat(Validator):

    __slots__ = ("options")

    def __init__(
        self,
        options=None,
        alias=None,
        replace=False,
    ):
        options = contracts.expect_container(
            self, "options", options, nullable=True, item_type=int
        )
        setattr = object.__setattr__
        setattr(self, "options", options)

        self._register(alias, replace)

    def __call__(self, value, __context=None):
        if not isinstance(value, float):
            if isinstance(value, int):
                # Always implicitly convert ``int`` to ``float``
                value = float(value)
        if self.options is not None and value not in self.options:
            raise OptionsError(expected=self.options, actual=value)
        return value


class ListValidationError(ValidationError):

    def __init__(self, *, context: t.Deque = None, **kw) -> None:
        super().__init__(context=context, **kw)
        self._errors = []

    def __len__(self) -> int:
        return len(self._errors)

    def add_context(self, node: t.Any) -> ValidationError:
        self._errors.append(node)
        return self

    def __repr__(self):
        return "\n".join([str(error) for error in self._errors])


class CheckValidator(object):

    def __init__(self, validators):
        self._validators = validators

    @classmethod
    def parse_from_yaml(cls, yaml_path, init=False):
        with open(yaml_path, mode='r') as f:
            state_config = yaml.safe_load(f)["state"]
        validators = []
        if init:
            for item in state_config.get("items", []):
                if "init" in item:
                    validators.append(InitFloat(options=[item["init"]]))
                else:
                    validators.append(InitFloat())
        else:
            for item in state_config.get("items", []):
                validators.append(Float(min=item.get("min", None), max=item.get("max", None)))
        return cls(validators)

    def validate(self, data):
        errors = ListValidationError()
        for i, (c, d) in enumerate(zip(self._validators, data)):
            try:
                c(d)
            except ValidationError as e:
                errors.add_context((i, e))
        if len(errors) > 0:
            raise errors



if __name__ == '__main__':
    init_validator = CheckValidator.parse_from_yaml("./static_scheme_0.yml", init=True)
    state_validator = CheckValidator.parse_from_yaml("./static_scheme_0.yml")
    s = [5, 1, 1, 8]
    try:
        init_validator.validate(s)
    except ListValidationError as e:
        print(e)
    print(s)
    try:
        state_validator.validate(s)
    except ListValidationError as e:
        print(e)
    with open("./static_scheme_0.yml", mode='r') as f:
        config = yaml.safe_load(f)
    print(config["trend"])
    print(type(config["trend"]))
