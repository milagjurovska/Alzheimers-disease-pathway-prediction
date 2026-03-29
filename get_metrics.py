import os, sys, warnings
warnings.filterwarnings('ignore')
sys.path.insert(0, os.path.dirname(__file__))
from features.feature_engineering import build_features
from models.evaluate import stratified_split
from sklearn.metrics import classification_report
from sklearn.ensemble import RandomForestClassifier
import xgboost as xgb
from sklearn.utils.class_weight import compute_sample_weight
import torch, torch.nn as nn, torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.preprocessing import StandardScaler
from models.neural_network import PathwayMLP, _compute_class_weights

X, y, feature_names, le = build_features(verbose=False)
X_train, X_test, y_train, y_test = stratified_split(X, y)

with open('data/results/class_reports.txt', 'w', encoding='utf-8') as f:
    # RF
    rf = RandomForestClassifier(class_weight="balanced", n_jobs=-1, random_state=42, n_estimators=330, max_depth=20, min_samples_leaf=1, max_features=0.5)
    rf.fit(X_train, y_train)
    y_pred = rf.predict(X_test)
    f.write("Random Forest\n======================\n")
    f.write(classification_report(y_test, y_pred, target_names=le.classes_, zero_division=0))
    f.write("\n\n")

    # XGBoost
    sw = compute_sample_weight("balanced", y_train)
    model = xgb.XGBClassifier(objective="multi:softprob", num_class=len(le.classes_), use_label_encoder=False, random_state=42, n_jobs=-1,
                              n_estimators=1000, max_depth=7, learning_rate=0.0587, subsample=0.828, colsample_bytree=0.57)
    model.fit(X_train, y_train, sample_weight=sw)
    y_pred = model.predict(X_test)
    f.write("XGBoost\n======================\n")
    f.write(classification_report(y_test, y_pred, target_names=le.classes_, zero_division=0))
    f.write("\n\n")

    # NN
    scaler = StandardScaler()
    X_train_s = scaler.fit_transform(X_train)
    X_test_s = scaler.transform(X_test)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    net = PathwayMLP(X.shape[1], len(le.classes_)).to(device)
    cw = _compute_class_weights(y_train, len(le.classes_)).to(device)
    crit = nn.CrossEntropyLoss(weight=cw)
    opt = optim.AdamW(net.parameters(), lr=1e-3, weight_decay=1e-4)
    dl = DataLoader(TensorDataset(torch.tensor(X_train_s, dtype=torch.float32), torch.tensor(y_train, dtype=torch.long)), batch_size=64, shuffle=True)
    
    for epoch in range(15):
        net.train()
        for X_b, y_b in dl:
            X_b, y_b = X_b.to(device), y_b.to(device)
            opt.zero_grad()
            loss = crit(net(X_b), y_b)
            loss.backward()
            nn.utils.clip_grad_norm_(net.parameters(), 5.0)
            opt.step()
            
    net.eval()
    with torch.no_grad():
        test_logits = net(torch.tensor(X_test_s, dtype=torch.float32).to(device))
        y_pred = test_logits.argmax(dim=1).cpu().numpy()
    
    f.write("Neural Network\n======================\n")
    f.write(classification_report(y_test, y_pred, target_names=le.classes_, zero_division=0))
