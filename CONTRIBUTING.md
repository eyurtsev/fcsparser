# Contributing

Thanks for your interest in contributing to the parser!

I no longer work on flow cytometry, so I will not personally develop any
new features; howevever, PRs are welcome. 

Be aware that I may comandeer the PRs and refactor them a bit.

## Setting up for development

The package uses [poetry](https://python-poetry.org/) together with
[poethepoet](https://github.com/nat-n/poethepoet).

### Install dependencies

```shell

poetry install --with dev,test
```

### List tasks

```shell
poe
```

### autoformat

```shell
poe autoformat
```

### unit tests 

```shell
poe pytest
```

