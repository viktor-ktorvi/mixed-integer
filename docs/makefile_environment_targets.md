# Environment and Installation Related Makefile Targets

## Poetry install targets

There are 2 possibilities available for how to manage `poetry` if it is not already
configured on your system.

**It is generally not recommended to install `poetry` in the same environment that will 
be managed by `poetry`; It is preferable to install it on the system using `pipx`, 
in order to minimize dependency conflicts.**

That being said, having `poetry` installed in a `conda` environment, and using `poetry` 
to manage that same `conda` environment is not the end of the world and is an acceptable 
workaround.

The following target will first try to install `Poetry` in the active `Conda` 
environment; if it can't find `Conda`, it will proceed to install via `pipx`

```shell
make poetry-install-auto
```

To install `Poetry` using `Conda`:

```shell
make conda-poetry-install
```

Using `pipx` will instead allow environment management directly with `Poetry.

```shell
make poetry-install
```

A virtual environment can then be created using the `make poetry-create-env`
command, and removed with the `make poetry-remove-env` command.

Information about the currently active environment used by Poetry, 
whether Conda or Poetry, can be seen using the `make poetry-env-info` command.

Both install methods can also be cleaned up:

```shell
make conda-poetry-uninstall
```
or
```shell
make poetry-uninstall
```

You can also use `make poetry-uninstall-pipx` or `make poetry-uninstall-venv` to also 
remove the `pipx` library and the `pipx` virtualenv, respectively, depending on how you 
chose to install `pipx` and `poetry`.

## Conda Targets

If you need or want to install Conda:
```shell
make conda-install 
```

To create the conda environment:
```shell
make conda-create-env
```

To remove the conda environment:
```shell
make conda-clean-env
```

Make sure to activate the configured environment before using the install targets.

## Install targets

**All `install` targets will first check if `Poetry` is available and try to install
it with the `make poetry-install-auto` target.**

To install the package, development dependencies and CLI tools (if available):
```shell
make install
```

To install only the package, without development tools:
```shell
make install-package
```
