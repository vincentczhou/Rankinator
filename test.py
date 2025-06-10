import hydra
import lightning
import torch
from omegaconf import DictConfig

from libs.utils import load_ckpt
from libs import (
    get_dataloaders,
)

torch.set_float32_matmul_precision('high')


@hydra.main(config_path="configs", config_name="train_v1", version_base="1.1")
def main(args: DictConfig):
    # model, model_args, tokenizer = load_ckpt(args.checkpoint_path, route_pickle=False)
    model, model_args, tokenizer = load_ckpt("/net/beliveau/vol1/home/vkchau/493g1/Rankinator/logs/2025-06-08/22-01-33/baseline.ckpt", route_pickle=False)
    # model, model_args, tokenizer = load_ckpt("/net/beliveau/vol1/home/vkchau/493g1/Rankinator/logs/2025-06-09/00-00-53/baselinenomapper.ckpt", route_pickle=False)

    _, val_dataloader = get_dataloaders(tokenizer, args)

    if args.compile:
        model.model = torch.compile(model.model)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.eval()
    y_preds = []
    y_trues = []
    for data in val_dataloader:
        res =  model(frames=data["frames"].to(device), decoder_input_ids=data["decoder_input_ids"].to(device))
        y_pred = torch.argmax(res.logits, dim=1)
        y_preds.extend(y_pred.tolist())
        y_trues.extend(data["labels"].tolist())

    import matplotlib.pyplot as plt
    from matplotlib.legend_handler import HandlerTuple
    import pandas as pd
    import seaborn as sns
    import numpy as np
    from sklearn.metrics import confusion_matrix

    classes = ("Unranked", "Ranked")
    cf_matrix = confusion_matrix(y_trues, y_preds)
    df_cm = pd.DataFrame(cf_matrix / np.sum(cf_matrix, axis=1)[:, None], index = [i for i in classes],
                        columns = [i for i in classes])
    sns.set_theme(rc={'figure.figsize':(14.4 , 9.6)})
    sns.set_theme(rc={"ps.fonttype": 42, "pdf.fonttype": 42, "svg.fonttype": "none"})
    sns.set_theme(context="paper", style="ticks", palette="muted", font_scale=2.5, rc={"axes.facecolor": "#EAEAF2", "lines.linewidth": 2})
    plt.figure(figsize = (12,7))
    sns.heatmap(df_cm, annot=True)
    plt.title("Baseline (No Mapper) Confusion Matrix")
    plt.savefig("baselinenoamppercf.png")
    from sklearn.metrics import f1_score
    print(f1_score(y_trues, y_preds))
    # trainer = lightning.Trainer(
    #     accelerator=args.device,
    #     precision=args.precision,
    # )

    # res = trainer.test(model, val_dataloader)


if __name__ == "__main__":
    main()
