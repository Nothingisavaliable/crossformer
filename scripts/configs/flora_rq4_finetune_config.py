import os

from ml_collections import ConfigDict
from ml_collections.config_dict import FieldReference

# Import registers the custom TFDS builder name "flora_rq4_crossformer".
import flora.baselines.crossformer.rlds_dataset  # noqa: F401

from crossformer.model.components.tokenizers import LowdimObsTokenizer
from crossformer.utils.spec import ModuleSpec


def get_config():
    mode = "full"
    task = "language_conditioned"
    head_name = "single_arm"

    rlds_root = os.environ.get("FLORA_RQ4_RLDS_ROOT", "")
    save_root = os.environ.get("FLORA_RQ4_CF_SAVE_DIR", "")

    dataset_kwargs = dict(
        name="flora_rq4_crossformer",
        data_dir=rlds_root,
        image_obs_keys={"primary": "image_primary"},
        proprio_obs_keys={"single_arm": "proprio"},
        proprio_obs_dims={"single_arm": 7},
        language_key="language_instruction",
        standardize_fn=None,
        action_proprio_normalization_type="normal",
        action_normalization_mask=[True, True, True, True, True, True, False],
    )

    # Pretrained CrossFormer has a primary image tokenizer and a 7D single-arm
    # action head, but no single-arm proprio tokenizer. Add only that missing
    # observation pathway; full finetuning updates it jointly with pretrained
    # parameters.
    update_config = dict(
        model=dict(
            observation_tokenizers=dict(
                flora_single_arm_proprio=ModuleSpec.create(
                    LowdimObsTokenizer,
                    obs_keys=["proprio_single_arm"],
                    dropout_rate=0.1,
                ),
            ),
        )
    )

    max_steps = FieldReference(20000)
    window_size = FieldReference(default=5)

    config = dict(
        update_config=update_config,
        pretrained_path="hf://rail-berkeley/crossformer",
        pretrained_step=None,
        batch_size=32,
        shuffle_buffer_size=20000,
        num_steps=max_steps,
        log_interval=100,
        eval_interval=1000,
        save_interval=1000,
        save_dir=save_root or None,
        seed=3407,
        wandb=dict(
            project="crossformer_finetune",
            group="flora_rq4",
            entity=None,
        ),
        dataset_kwargs=dataset_kwargs,
        modality=task,
        finetuning_mode=mode,
        head_name=head_name,
        window_size=window_size,
        optimizer=dict(
            learning_rate=dict(
                name="cosine",
                init_value=0.0,
                peak_value=1e-4,
                warmup_steps=1000,
                decay_steps=max_steps,
                end_value=0.0,
            ),
            weight_decay=0.01,
            clip_gradient=1.0,
            frozen_keys=None,
            grad_accumulation_steps=None,
        ),
        val_kwargs=dict(
            val_shuffle_buffer_size=2000,
            num_val_batches=16,
        ),
    )

    config["traj_transform_kwargs"] = dict(
        window_size=window_size,
        action_horizon=4,
        goal_relabeling_strategy=None,
        task_augment_strategy="delete_task_conditioning",
        task_augment_kwargs=dict(keep_image_prob=0.0),
    )

    workspace_augment_kwargs = dict(
        random_resized_crop=dict(scale=[0.8, 1.0], ratio=[0.9, 1.1]),
        random_brightness=[0.1],
        random_contrast=[0.9, 1.1],
        random_saturation=[0.9, 1.1],
        random_hue=[0.05],
        augment_order=[
            "random_resized_crop",
            "random_brightness",
            "random_contrast",
            "random_saturation",
            "random_hue",
        ],
    )

    config["frame_transform_kwargs"] = dict(
        resize_size={"primary": (224, 224)},
        image_augment_kwargs=dict(primary=workspace_augment_kwargs),
    )
    config["frame_transform_threads"] = 16
    return ConfigDict(config)
