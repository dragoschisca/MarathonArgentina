# 💰 Corectarea Data Leakage în Predicția Neplății Creditului

## 🎯 Scop

Scopul proiectului este de a elimina **scurgerile de date (data leakage)** dintr-un set de date privind creditele, pentru a asigura că modelul de machine learning învață doar din informațiile disponibile **înainte** de momentul deciziei de creditare.

---

## ⚙️ Ce face scriptul

1. 🔁 Elimină înregistrările duplicate.
2. 🔒 Elimină coloana `last_audit_team_id`, care conține informații generate **după** acordarea creditului.
3. ⚖️ Aplică standardizarea (`StandardScaler`) doar pe datele de antrenare — pentru a evita scurgerea statistică.
4. ➕ Creează două variabile noi sigure:

   * `debt_to_income_ratio` — raportul dintre datorie și venit
   * `loan_term_risk` — scor estimat al riscului în funcție de durata creditului
5. 💾 Salvează un fișier curat: `loan_data_preprocessed.csv`

---

## ▶️ Cum se rulează

În terminal:

```bash
python fix_leakage.py
```

---

## 📂 Rezultat

Fișierul final **loan_data_preprocessed.csv** conține:

* Toate coloanele sigure din setul inițial
* Cele două coloane noi create
* Fără duplicate și fără scurgeri de date

---

## ✅ Probleme corectate

| Tip scurgere     | Descriere                                                         | Soluție                         |
| ---------------- | ----------------------------------------------------------------- | ------------------------------- |
| Duplicate        | Înregistrări repetate care distorsionau modelul                   | Eliminare completă              |
| Temporal leakage | Coloana `last_audit_team_id` conținea informații apărute ulterior | Eliminată                       |
| Scaling leakage  | Standardizarea aplicată înainte de împărțirea datelor             | Corectat: `fit` doar pe `train` |

