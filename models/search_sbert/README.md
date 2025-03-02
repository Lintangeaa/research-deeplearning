---
tags:
- sentence-transformers
- sentence-similarity
- feature-extraction
- generated_from_trainer
- dataset_size:182
- loss:CosineSimilarityLoss
base_model: sentence-transformers/multi-qa-MiniLM-L6-dot-v1
widget:
- source_sentence: Perbedaan Fleece dan Baby terry ?
  sentences:
  - Ya, kami menyediakan asuransi pengiriman dari pihak ketiga. Namun, untuk biaya
    asuransi akan ditanggung oleh pihak customer.
  - "Kedua kain ini sebenarnya diperuntukan untuk jaket atau hoodie, hanya memang\
    \ terdapat perbedaannya. \nUntuk fleece : mempunyai gramasi agak berat / tebal\
    \ (sekitar 280-290 gsm), biasa setting nya finish, dan untuk tesktur di balik\
    \ kain nya berupa bulu bulu\nUntuk babyterry : biasanya mempunyai gramasi lebih\
    \ ringan (sekitar 230 gsm), biasa setting nya finish atau tubular, tekstur di\
    \ balik kain nya berbentuk seperti loop.\n"
  - Jika dibiarkan lama (lebih dari 3 bulan) dan tidak dijahit menjadi baju dan dicuci.
    Maka kain akan menjadi mudah hancur dan robek.
- source_sentence: Kalau beli di bawah 5kg per item berapa?
  sentences:
  - 'Kak  mengenai kain combed bisa menggunakan mesin apapun, Biasanya untuk skala
    besar ( garment) akan lebih efektif langsung menggunakan mesin obras karena lebih
    cepat dan kuat ( 3 benang ). namun dengan mesin jahit biasa pun bisa ( biasanya
    1 benang ) '
  - Jenis kain dengan tekstur unik dengan benang yang timbul dan tenggelam
  - Untuk harga kain tergantung jenis dan warna kainnya kak. Bisa diinfokan kebutuhannya
    di kain apa kak agar kami bisa memberikan pricelistnya
- source_sentence: 'Refund berapa hari '
  sentences:
  - 'Terima kasih telah menunggu, berikut ini foto resinya:


    *kirim foto*                                                                                                                                                           Kami
    ingin menginformasikan, jika ingin melihat foto resi/ no resi lebih cepat saat
    ini sudah bisa melalui portal.knitto.co.id > status order > pilih no order. Silakan
    dicoba '
  - 'Cotton bamboo permukaan kainnya cenderung lebih berbulu karena bahan dasarnya,
    sehingga kami lebih merekomendasikan cotton combed dan cotton modal ya Kak. '
  - "Kak, Kami informasikan untuk proses refund \nBank BCA 3-4 hari kerja \nBank (MANDIRI,\
    \ BRI, dll ) 4-7 hari kerja \n\nsyarat dan ketentuan refund, diantaranya:\n1.\
    \ KTP (yang mengirimkan transfer)\n2. Bukti Transfer (beserta nama pemilik rekening\
    \ saat melakukan transfer)\n3. Nomor rekening pemilik"
- source_sentence: Apa itu Threetone & Twotone?
  sentences:
  - 'Kakak tetap dapat melakukan order melalui whatsapp namun jika antrian chat sedang
    padat, proses order akan lebih cepat menggunakan website customer portal '
  - 'Threetone: Kain dengan 3 warna berbeda dalam satu kain.

    Twotone: Kain dengan 2 warna berbeda dalam satu kain'
  - 'Mohon mengisi data sebagai pelengkap pengantaran '
- source_sentence: Apakah ada kartu nama Knitto ?
  sentences:
  - '"[{\"name\": \"HOLIS\", \"lokasi\": \"Bandung\"}, {\"name\": \"HOS COKROAMINOTO\",
    \"lokasi\": \"Jogja\"}, {\"name\": \"KEBON JUKUT\", \"lokasi\": \"Bandung\"},
    {\"name\": \"SOEKARNO\", \"lokasi\": \"Surabaya\"}, {\"name\": \"SUDIRMAN\", \"lokasi\":
    \"Semarang\"}]"'
  - "TENCEL™ Modal blended with Cotton\nStruktur: Single Knit\nKetebalan: 30s\nKomposisi:\
    \ 50% TENCEL™ Modal 50% Cotton\nGramasi: 150-160 gsm\nLebar: 42”\nProduct Knowledge\n\
    TENCEL™ Modal merupakan bahan yang dapat diuraikan secara alami, dan juga dapat\
    \ dibuat menjadi kompos sehingga sepenuhnya dapat kembali ke alam.\nSerat Modal\
    \ secara alami dapat mengatur perpindahan kelembaban, meningkatkan kualitas dari\
    \ kain dan terasa nyaman di kulit. \nSub-struktur serat modal disusun untuk dapat\
    \ mengatur penyerapan dan pelepasan uap air / kelembaban. Kain ini mempunyai sifat\
    \ alami sebagai pengatur suhu alami tubuh, dan memberikan sensasi dingin menyegarkan\
    \ pada kulit.\nWarna yang terkandung pada serat kain TENCEL ™ Modal diserap dengan\
    \ baik sehingga kecerahan warna lebih tahan lama dibandingkan dengan serat kain\
    \ yang diwarnai secara konvensional, warna pada kain tidak mudah pudar bahkan\
    \ setelah dicuci berulang kali.\nSerat TENCEL ™ Modal menawarkan kualitas kain\
    \ yang sangat lembut dan juga tahan lama dan mempunyai fleksibilitas yang tinggi.\
    \ Hal ini karena TENCEL ™ Modal mempunyai penampang serat yang ramping atau tipis\
    \ sehingga kain terasa lembut bahkan setelah dicuci berulang kali. Lembutnya kain\
    \ TENCEL ™ Modal terasa dua kali lebih lembut dibandingkan dengan serat kapas\
    \ dan tetap terjaga kualitasnya bahkan setelah proses pencucian dan pengeringan\
    \ berulang kali.\nSerat TENCEL ™ diproduksi oleh Perusahaan Lenzing dari Austria.\
    \ \nMerk TENCEL ™ sudah dikenal secara internasional dan digunakan oleh brand-brand\
    \ clothing ternama di seluruh dunia\n"
  - Berikut untuk kartu namanya ya kak https://drive.google.com/file/d/1SpZTbNhbkRF4RDczuHBGgccM1j8Z_Jtr/view?usp=share_link
pipeline_tag: sentence-similarity
library_name: sentence-transformers
---

# SentenceTransformer based on sentence-transformers/multi-qa-MiniLM-L6-dot-v1

This is a [sentence-transformers](https://www.SBERT.net) model finetuned from [sentence-transformers/multi-qa-MiniLM-L6-dot-v1](https://huggingface.co/sentence-transformers/multi-qa-MiniLM-L6-dot-v1). It maps sentences & paragraphs to a 384-dimensional dense vector space and can be used for semantic textual similarity, semantic search, paraphrase mining, text classification, clustering, and more.

## Model Details

### Model Description
- **Model Type:** Sentence Transformer
- **Base model:** [sentence-transformers/multi-qa-MiniLM-L6-dot-v1](https://huggingface.co/sentence-transformers/multi-qa-MiniLM-L6-dot-v1) <!-- at revision 4151c507ffb0f2fcd311cf431f54b5fc7d097851 -->
- **Maximum Sequence Length:** 512 tokens
- **Output Dimensionality:** 384 dimensions
- **Similarity Function:** Dot Product
<!-- - **Training Dataset:** Unknown -->
<!-- - **Language:** Unknown -->
<!-- - **License:** Unknown -->

### Model Sources

- **Documentation:** [Sentence Transformers Documentation](https://sbert.net)
- **Repository:** [Sentence Transformers on GitHub](https://github.com/UKPLab/sentence-transformers)
- **Hugging Face:** [Sentence Transformers on Hugging Face](https://huggingface.co/models?library=sentence-transformers)

### Full Model Architecture

```
SentenceTransformer(
  (0): Transformer({'max_seq_length': 512, 'do_lower_case': False}) with Transformer model: BertModel 
  (1): Pooling({'word_embedding_dimension': 384, 'pooling_mode_cls_token': True, 'pooling_mode_mean_tokens': False, 'pooling_mode_max_tokens': False, 'pooling_mode_mean_sqrt_len_tokens': False, 'pooling_mode_weightedmean_tokens': False, 'pooling_mode_lasttoken': False, 'include_prompt': True})
)
```

## Usage

### Direct Usage (Sentence Transformers)

First install the Sentence Transformers library:

```bash
pip install -U sentence-transformers
```

Then you can load this model and run inference.
```python
from sentence_transformers import SentenceTransformer

# Download from the 🤗 Hub
model = SentenceTransformer("sentence_transformers_model_id")
# Run inference
sentences = [
    'Apakah ada kartu nama Knitto ?',
    'Berikut untuk kartu namanya ya kak https://drive.google.com/file/d/1SpZTbNhbkRF4RDczuHBGgccM1j8Z_Jtr/view?usp=share_link',
    '"[{\\"name\\": \\"HOLIS\\", \\"lokasi\\": \\"Bandung\\"}, {\\"name\\": \\"HOS COKROAMINOTO\\", \\"lokasi\\": \\"Jogja\\"}, {\\"name\\": \\"KEBON JUKUT\\", \\"lokasi\\": \\"Bandung\\"}, {\\"name\\": \\"SOEKARNO\\", \\"lokasi\\": \\"Surabaya\\"}, {\\"name\\": \\"SUDIRMAN\\", \\"lokasi\\": \\"Semarang\\"}]"',
]
embeddings = model.encode(sentences)
print(embeddings.shape)
# [3, 384]

# Get the similarity scores for the embeddings
similarities = model.similarity(embeddings, embeddings)
print(similarities.shape)
# [3, 3]
```

<!--
### Direct Usage (Transformers)

<details><summary>Click to see the direct usage in Transformers</summary>

</details>
-->

<!--
### Downstream Usage (Sentence Transformers)

You can finetune this model on your own dataset.

<details><summary>Click to expand</summary>

</details>
-->

<!--
### Out-of-Scope Use

*List how the model may foreseeably be misused and address what users ought not to do with the model.*
-->

<!--
## Bias, Risks and Limitations

*What are the known or foreseeable issues stemming from this model? You could also flag here known failure cases or weaknesses of the model.*
-->

<!--
### Recommendations

*What are recommendations with respect to the foreseeable issues? For example, filtering explicit content.*
-->

## Training Details

### Training Dataset

#### Unnamed Dataset

* Size: 182 training samples
* Columns: <code>sentence_0</code>, <code>sentence_1</code>, and <code>label</code>
* Approximate statistics based on the first 182 samples:
  |         | sentence_0                                                                        | sentence_1                                                                         | label                        |
  |:--------|:----------------------------------------------------------------------------------|:-----------------------------------------------------------------------------------|:-----------------------------|
  | type    | string                                                                            | string                                                                             | int                          |
  | details | <ul><li>min: 2 tokens</li><li>mean: 16.63 tokens</li><li>max: 49 tokens</li></ul> | <ul><li>min: 3 tokens</li><li>mean: 97.12 tokens</li><li>max: 512 tokens</li></ul> | <ul><li>1: 100.00%</li></ul> |
* Samples:
  | sentence_0                                                      | sentence_1                                                                                                                                                                                                           | label          |
  |:----------------------------------------------------------------|:---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|:---------------|
  | <code>Apakah ada kartu nama Knitto ?</code>                     | <code>Berikut untuk kartu namanya ya kak https://drive.google.com/file/d/1SpZTbNhbkRF4RDczuHBGgccM1j8Z_Jtr/view?usp=share_link</code>                                                                                | <code>1</code> |
  | <code>Apakah Knitto punya pabrik/ produksi kain sendiri?</code> | <code>Untuk Knitto hanya toko tetapi group perusahaan kami memproduksi kainnya sendiri sehingga lebih terjamin kualitasnya karena langsung dari pabrik.</code>                                                       | <code>1</code> |
  | <code>Boleh price list nya?</code>                              | <code>Pricelist Standard & Pricelist + Color Category<br>Harga yang tertera adalah harga per kg. 1 roll setara dengan 25 kg bruto.<br>Jika ada yang kurang dipahami mohon ditanyakan, kami siap membantu kak </code> | <code>1</code> |
* Loss: [<code>CosineSimilarityLoss</code>](https://sbert.net/docs/package_reference/sentence_transformer/losses.html#cosinesimilarityloss) with these parameters:
  ```json
  {
      "loss_fct": "torch.nn.modules.loss.MSELoss"
  }
  ```

### Training Hyperparameters
#### Non-Default Hyperparameters

- `per_device_train_batch_size`: 16
- `per_device_eval_batch_size`: 16
- `num_train_epochs`: 1
- `multi_dataset_batch_sampler`: round_robin

#### All Hyperparameters
<details><summary>Click to expand</summary>

- `overwrite_output_dir`: False
- `do_predict`: False
- `eval_strategy`: no
- `prediction_loss_only`: True
- `per_device_train_batch_size`: 16
- `per_device_eval_batch_size`: 16
- `per_gpu_train_batch_size`: None
- `per_gpu_eval_batch_size`: None
- `gradient_accumulation_steps`: 1
- `eval_accumulation_steps`: None
- `torch_empty_cache_steps`: None
- `learning_rate`: 5e-05
- `weight_decay`: 0.0
- `adam_beta1`: 0.9
- `adam_beta2`: 0.999
- `adam_epsilon`: 1e-08
- `max_grad_norm`: 1
- `num_train_epochs`: 1
- `max_steps`: -1
- `lr_scheduler_type`: linear
- `lr_scheduler_kwargs`: {}
- `warmup_ratio`: 0.0
- `warmup_steps`: 0
- `log_level`: passive
- `log_level_replica`: warning
- `log_on_each_node`: True
- `logging_nan_inf_filter`: True
- `save_safetensors`: True
- `save_on_each_node`: False
- `save_only_model`: False
- `restore_callback_states_from_checkpoint`: False
- `no_cuda`: False
- `use_cpu`: False
- `use_mps_device`: False
- `seed`: 42
- `data_seed`: None
- `jit_mode_eval`: False
- `use_ipex`: False
- `bf16`: False
- `fp16`: False
- `fp16_opt_level`: O1
- `half_precision_backend`: auto
- `bf16_full_eval`: False
- `fp16_full_eval`: False
- `tf32`: None
- `local_rank`: 0
- `ddp_backend`: None
- `tpu_num_cores`: None
- `tpu_metrics_debug`: False
- `debug`: []
- `dataloader_drop_last`: False
- `dataloader_num_workers`: 0
- `dataloader_prefetch_factor`: None
- `past_index`: -1
- `disable_tqdm`: False
- `remove_unused_columns`: True
- `label_names`: None
- `load_best_model_at_end`: False
- `ignore_data_skip`: False
- `fsdp`: []
- `fsdp_min_num_params`: 0
- `fsdp_config`: {'min_num_params': 0, 'xla': False, 'xla_fsdp_v2': False, 'xla_fsdp_grad_ckpt': False}
- `fsdp_transformer_layer_cls_to_wrap`: None
- `accelerator_config`: {'split_batches': False, 'dispatch_batches': None, 'even_batches': True, 'use_seedable_sampler': True, 'non_blocking': False, 'gradient_accumulation_kwargs': None}
- `deepspeed`: None
- `label_smoothing_factor`: 0.0
- `optim`: adamw_torch
- `optim_args`: None
- `adafactor`: False
- `group_by_length`: False
- `length_column_name`: length
- `ddp_find_unused_parameters`: None
- `ddp_bucket_cap_mb`: None
- `ddp_broadcast_buffers`: False
- `dataloader_pin_memory`: True
- `dataloader_persistent_workers`: False
- `skip_memory_metrics`: True
- `use_legacy_prediction_loop`: False
- `push_to_hub`: False
- `resume_from_checkpoint`: None
- `hub_model_id`: None
- `hub_strategy`: every_save
- `hub_private_repo`: None
- `hub_always_push`: False
- `gradient_checkpointing`: False
- `gradient_checkpointing_kwargs`: None
- `include_inputs_for_metrics`: False
- `include_for_metrics`: []
- `eval_do_concat_batches`: True
- `fp16_backend`: auto
- `push_to_hub_model_id`: None
- `push_to_hub_organization`: None
- `mp_parameters`: 
- `auto_find_batch_size`: False
- `full_determinism`: False
- `torchdynamo`: None
- `ray_scope`: last
- `ddp_timeout`: 1800
- `torch_compile`: False
- `torch_compile_backend`: None
- `torch_compile_mode`: None
- `dispatch_batches`: None
- `split_batches`: None
- `include_tokens_per_second`: False
- `include_num_input_tokens_seen`: False
- `neftune_noise_alpha`: None
- `optim_target_modules`: None
- `batch_eval_metrics`: False
- `eval_on_start`: False
- `use_liger_kernel`: False
- `eval_use_gather_object`: False
- `average_tokens_across_devices`: False
- `prompts`: None
- `batch_sampler`: batch_sampler
- `multi_dataset_batch_sampler`: round_robin

</details>

### Framework Versions
- Python: 3.12.6
- Sentence Transformers: 3.4.1
- Transformers: 4.49.0
- PyTorch: 2.6.0
- Accelerate: 1.4.0
- Datasets: 3.3.2
- Tokenizers: 0.21.0

## Citation

### BibTeX

#### Sentence Transformers
```bibtex
@inproceedings{reimers-2019-sentence-bert,
    title = "Sentence-BERT: Sentence Embeddings using Siamese BERT-Networks",
    author = "Reimers, Nils and Gurevych, Iryna",
    booktitle = "Proceedings of the 2019 Conference on Empirical Methods in Natural Language Processing",
    month = "11",
    year = "2019",
    publisher = "Association for Computational Linguistics",
    url = "https://arxiv.org/abs/1908.10084",
}
```

<!--
## Glossary

*Clearly define terms in order to be accessible across audiences.*
-->

<!--
## Model Card Authors

*Lists the people who create the model card, providing recognition and accountability for the detailed work that goes into its construction.*
-->

<!--
## Model Card Contact

*Provides a way for people who have updates to the Model Card, suggestions, or questions, to contact the Model Card authors.*
-->