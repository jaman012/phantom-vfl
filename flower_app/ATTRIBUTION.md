# Attribution

The Flower app in this folder (`flower_app/`) is a modified version of an example of the **Flower** federated learning framework. Flower itself is a dependency; it is not part of this repository.

## Original work

| Item | Value |
|---|---|
| Project | Flower (https://flower.ai) |
| Repository | https://github.com/adap/flower |
| Example | `examples/vertical-fl` (vertical federated learning on the Titanic dataset) |
| Source version | The example was taken from the main branch of the repository after the v1.25.0 release (16 Dec 2025), on 26 January 2026 (date given by the author). Its code is identical to the example at tags `v1.25.0` and `v1.26.0` (checked by MD5 against the files at the tags). The exact commit is not recorded. |
| Licence | Apache License, Version 2.0. The file `LICENSE` in this folder is the `LICENSE` file of the repository at tag `v1.26.0`, unchanged. |
| Copyright notice | The upstream `LICENSE` file, the README files and the example files do not name a copyright owner. The example files have no copyright header. |
| NOTICE file | The repository has no `NOTICE` file at tag `v1.26.0`, so there is no NOTICE text to carry. |

## What was taken

The structure of the app (`pyproject.toml`, `vertical_fl/client_app.py`, `vertical_fl/server_app.py`, `vertical_fl/task.py`), the message handlers `generate_embeddings` and `apply_gradients`, the server training loop, the functions `send_gradients_to_clients` and `get_remote_embeddings`, the client model (`ClientModel`), the server (head) model (`ServerModel`) and `evaluate_head_model`.

## What was changed

Each modified file has a change notice at the top (Apache License 2.0, section 4(b)). `MODIFICATIONS.md` lists every change file by file. The data loading, the loss functions, the test metrics and the results folder are the author's own code in `framework/training/` of this repository.

## Citation

The Flower README (tag `v1.26.0`) asks users to cite Flower as follows:

```bibtex
@article{beutel2020flower,
  title={Flower: A Friendly Federated Learning Research Framework},
  author={Beutel, Daniel J and Topal, Taner and Mathur, Akhil and Qiu, Xinchi and Fernandez-Marques, Javier and Gao, Yan and Sani, Lorenzo and Kwing, Hei Li and Parcollet, Titouan and Gusmão, Pedro PB de and Lane, Nicholas D},
  journal={arXiv preprint arXiv:2007.14390},
  year={2020}
}
```

## Acknowledgement

Thanks to the Flower team for the framework and for the vertical federated learning example.

See also `docs/flower_reference.md` in the repository. This file is a record of facts, not legal advice.
