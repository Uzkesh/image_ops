import click

from handlers.data_packer.packer import cli_packing
from handlers.data_packer.unpacker import cli_unpacking
from handlers.prepare_data import cli_prepare_data

cli = click.CommandCollection(
    sources=[
        cli_packing,
        cli_unpacking,
        cli_prepare_data,
    ]
)


if __name__ == "__main__":
    # Примеры вызова:
    # python manage.py packing -s /some/dir/with/files
    # python manage.py unpacking -s /some/dir/with/files
    # python manage.py prepare-data -r -s /home/some_dataset -lt 512 -w 8
    cli()
