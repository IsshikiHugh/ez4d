import rich
import traceback

from typing import Optional


__all__ = ['show_call_stacks']


def show_call_stacks(
    base_path: Optional[str] = None,
    ignore_lib_keyword: Optional[str] = 'miniconda',
) -> None:

    stacks = traceback.extract_stack()
    stacks = [s for s in stacks if ignore_lib_keyword not in s.filename][:-2]

    # Hierarchical representation
    for s in stacks:
        if base_path is not None:
            fn = s.filename.replace(base_path, '')
        else:
            fn = s.filename
        rich.print(
            f'[bold blue]{fn}[/bold blue]:[green]{s.lineno}[/green] in [yellow]{s.name}[/yellow]'
        )
        content = s.line
        if content == '':
            content = '<empty>'
        rich.print(f'    [bright_black]{content}[/bright_black]')