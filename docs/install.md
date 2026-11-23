# Install HiData

```{contents}
:local:
:depth: 3
```

## PyPI

HiData needs Python 3.12 or newer. Install it from [PyPI](https://pypi.org/project/hidata/):

```console
pip install hidata
```

## Development version

The latest version on [GitHub](https://github.com/theislab/hidata) is installed with:

```console
pip install git+https://github.com/theislab/hidata
```

## Optional dependencies

The PyTorch dataset in {mod}`hidata.ml` needs the `ml` extra:

```console
pip install "hidata[ml]"
```

Group or subgroup levels stored as an EHRData need [ehrdata](https://ehrdata.readthedocs.io) 0.6 or later, which is
installed with the `ehr` extra:

```console
pip install "hidata[ehr]"
```

## Troubleshooting

If there are issues that have not been described, addressed, or documented, please consider
[opening an issue](https://github.com/theislab/hidata/issues).
