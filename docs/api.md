# API

```{eval-rst}
.. module:: hidata
```

The central class:

```{eval-rst}
.. autosummary::
   :toctree: generated/

   HiData
```

(combining-api)=

## Combining

Combining {class}`HiData` objects.

```{eval-rst}
.. autosummary::
   :toctree: generated/

   concat
```

(reading-api)=

## Reading

Reading HiData's native formats `.hd.h5` and `.hd.zarr`, and files and stores written by donordata.
(These functions are also exported as {func}`read_hd`, {func}`read_h5_hd` and {func}`read_zarr_hd`.)

```{eval-rst}
.. autosummary::
   :toctree: generated/

   io.read_hd
   io.read_h5_hd
   io.read_zarr_hd
```

(writing-api)=

## Writing

Writing a complete {class}`HiData` object to disk.

```{eval-rst}
.. autosummary::

   HiData.write_hd
   HiData.write_h5_hd
   HiData.write_zarr_hd
```

## Validation

Checking an object against the {doc}`specification <io/spec>`.

```{eval-rst}
.. autosummary::
   :toctree: generated/

   validate
   HiDataSchemaError
```

## Extensions

```{eval-rst}
.. autosummary::
   :toctree: generated/

   register_hidata_namespace
```

## Preprocessing

```{eval-rst}
.. autosummary::
   :toctree: generated/

   pp.cell_level_obs_filter
   pp.group_level_obs_filter
   pp.group_level_var_filter
```

## Machine learning

Requires PyTorch, see {doc}`install`.

```{eval-rst}
.. autosummary::
   :toctree: generated/

   ml.MILDataset
   ml.mil_collate_fn
```
