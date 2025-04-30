Please, use scripts from HoTPP to prepare the dataset or link the HoTPP data dir to this folder.

# Example training
To train a next-event prediction model use the following command:
```bash
python3 -m hotpp.train --config-dir configs --config-name next_item.yaml
```

To fine-tune the model in a supervised mode:
```bash
python3 -m hotpp.train --config-dir configs --config-name rnn_ft.yaml base_name=next_item
```

To disable downstream evaluation with LightGBM use "test_downstream=false" parameter.
