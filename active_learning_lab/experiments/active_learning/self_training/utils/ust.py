from torch.nn import Dropout


def change_dropout_rate(model, dropout=0.1, hidden_dropout=0.1, attention_dropout=0.1):
    if hasattr(model, 'model_body'):
        modules = dict(model.model_body.named_modules())
    else:
        modules = dict(model.named_modules())

    # 'dropout' is assumed to exist for BERT-like models, but not for SetFit with regression
    if 'dropout' in modules:
        modules['dropout'].p = dropout

    for name, module in modules.items():
        if 'layer' in name and isinstance(module, Dropout):
            if 'attention' in name:
                module.p = attention_dropout
            else:
                module.p = hidden_dropout
