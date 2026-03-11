import importlib.abc
import importlib.util
from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parent
SRC = ROOT / "src"

if SRC.is_dir():
    src_str = str(SRC)
    if src_str not in sys.path:
        sys.path.insert(0, src_str)


class _LocalLeRobotFinder(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname: str, path=None, target=None):
        if fullname != "lerobot":
            return None

        package_root = SRC / "lerobot"
        init_file = package_root / "__init__.py"
        if not init_file.is_file():
            return None

        return importlib.util.spec_from_file_location(
            fullname,
            init_file,
            submodule_search_locations=[str(package_root)],
        )


if SRC.is_dir():
    sys.meta_path.insert(0, _LocalLeRobotFinder())
    spec = _LocalLeRobotFinder().find_spec("lerobot")
    if spec is not None and "lerobot" not in sys.modules:
        module = importlib.util.module_from_spec(spec)
        sys.modules["lerobot"] = module
        assert spec.loader is not None
        spec.loader.exec_module(module)
