#!/usr/bin/env python
"""CLI tai lap bai "Tap du doan conformal theo muc do..." (xem README.md).

Sinh lai cac bang cua bai tu file du doan co san (khong can anh, khong can GPU):
    python scripts/bench.py validate        # kiem chung cac file du doan
    python scripts/bench.py hieu-chinh      # Bang 2-11: 9 mo hinh luot 1-2
    python scripts/bench.py hieu-chinh --cap-do    # Bang 12: ResNet18/50/152 (luot 3)
    python scripts/bench.py khong-chac      # lech theo dai, hieu co tap theo cap
    python scripts/bench.py kiem-dinh-bai   # Holm cho cap co tap cung luot, nguong +inf

Chuan bi du lieu de huan luyen lai (can RoCoLe goc, xem README.md):
    python scripts/bench.py doctor          # kiem tra moi thu san sang chua
    python scripts/bench.py index           # dung bang chi muc anh + khung bao la
    python scripts/bench.py splits          # chia 5 fold theo cay, DONG BANG (da co san)
    python scripts/bench.py images          # cat + thu nho -> PNG 224 (mot lan)
    python scripts/bench.py package         # dong goi mang len Colab

Thu tu bat buoc: doctor -> index -> splits -> images -> package.
Sau khi `splits` chay xong VA file duoc commit, dung chay lai no: sinh lai split
nghia la moi ket qua train truoc do khong con so sanh duoc voi sau do.
"""

from __future__ import annotations

import argparse
import json
import sys
import zipfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import pandas as pd

from coffeeguard.bench import contract as ct
from coffeeguard.bench import index as bidx
from coffeeguard.bench import paths as P
from coffeeguard.bench import predictions as bpred
from coffeeguard.bench import preprocess as bprep
from coffeeguard.bench import splits as bsplit

pd.set_option("display.width", 200)
pd.set_option("display.max_columns", 50)


def _tieu_de(s: str) -> None:
    print("\n" + "=" * 78 + "\n" + s + "\n" + "=" * 78)


def _an_toan(s: str) -> str:
    return "".join(c if c.isalnum() or c in "-_." else "_" for c in s)


# --------------------------------------------------------------------- doctor
def cmd_doctor(_args) -> int:
    _tieu_de("DOCTOR - nguon du lieu")
    can = {
        "RoCoLe anh": (P.ROCOLE_PHOTOS, "*.jpg", 1560),
        "RoCoLe COCO (mat na la)": (P.ROCOLE_COCO, None, None),
        "RoCoLe nhan da chuan hoa": (P.ROCOLE_LABELS, None, None),
    }
    thieu = 0
    for ten, (p, glob, mong) in can.items():
        if not p.exists():
            print(f"  [THIEU] {ten:32s} {p}")
            thieu += 1
            continue
        if glob:
            n = sum(1 for _ in p.glob(glob))
            dau = "OK  " if (mong is None or n == mong) else "CHUY"
            extra = f" (mong doi {mong})" if mong else ""
            print(f"  [{dau}] {ten:32s} {n} file{extra}")
        else:
            print(f"  [OK  ] {ten:32s} {p.stat().st_size / 1024:.0f} KB")

    _tieu_de("DOCTOR - tao pham da dong bang")
    for ten, p in [("index", P.INDEX_CSV), ("split 5 fold", P.FOLDS_JSON)]:
        print(f"  [{'OK  ' if p.exists() else 'CHUA'}] {ten:32s} {p}")
    if P.PROCESSED.exists():
        for d in sorted(P.PROCESSED.iterdir()):
            if d.is_dir():
                n = sum(1 for _ in d.glob("*.png"))
                print(f"  [OK  ] anh da xu ly {d.name:20s} {n} PNG")

    _tieu_de("DOCTOR - thu vien")
    for ten in ("numpy", "pandas", "PIL", "sklearn", "torch", "timm", "onnx", "onnxruntime"):
        try:
            m = __import__(ten)
            print(f"  [OK  ] {ten:12s} {getattr(m, '__version__', '?')}")
        except ImportError:
            bat_buoc = ten in ("numpy", "pandas", "PIL", "sklearn")
            ghi = "" if bat_buoc else "  (chi can khi train - chay tren Colab)"
            print(f"  [{'THIEU' if bat_buoc else 'khong'}] {ten:12s}{ghi}")

    if P.FOLDS_JSON.exists():
        meta = bsplit.doc_split()
        if meta.get("index_sha256") and P.INDEX_CSV.exists():
            khop = meta["index_sha256"] == P.bam_file(P.INDEX_CSV)
            print(f"\n  index khop split da dong bang: {'CO' if khop else 'KHONG - DUNG LAI'}")
            if not khop:
                print("    Index da doi sau khi split duoc dong bang. Hoac khoi phuc index,")
                print("    hoac chap nhan sinh lai split VA train lai TAT CA model.")
                return 1
    print(f"\n  {thieu} nguon con thieu.")
    return 0


# ---------------------------------------------------------------------- index
def cmd_index(_args) -> int:
    _tieu_de("DUNG BANG CHI MUC")
    kq = bidx.dung_index()
    df = kq.df
    dich = bidx.ghi_index(kq)
    print(f"  {len(df)} anh - {df.plant_id.nunique()} cay - {df.leaf_id.nunique()} la")
    khong_nhan = kq.bo_qua["co_anh_khong_nhan"]
    if khong_nhan:
        print(f"  {len(khong_nhan)} anh CO file nhung KHONG co nhan -> khong dung: {khong_nhan[:3]}")

    print("\n-- phan bo 6 lop --")
    d6 = df.label6.value_counts().reindex(list(ct.CLASSES_6)).fillna(0).astype(int)
    for c in ct.CLASSES_6:
        n = int(d6[c])
        print(f"   {c:18s} {n:5d}  ({n / len(df) * 100:5.1f}%)")
    print(f"   -> doan 'healthy' moi luc da duoc {int(d6['healthy']) / len(df) * 100:.1f}% accuracy.")
    print("      Accuracy khong duoc dung lam chi so chinh.")

    print("\n-- ti le pixel LA (p25 / trung vi / p75) --")
    for ten, cot in (("trong anh goc 16:9", "leaf_frac_khung"),
                     ("sau letterbox  <- model thuc nhan", "leaf_frac_letterbox")):
        q = df[cot].quantile([0.25, 0.5, 0.75])
        print(f"   {ten:36s} {q[0.25]*100:5.1f}% / {q[0.5]*100:5.1f}% / {q[0.75]*100:5.1f}%")
    print(f"   khung bao la trung vi {df.bbox_ti_le.median():.2f}:1 -> nhoi vao o vuong se")
    print("   sinh vien, do la ly do dung cua so vuong quanh la thay vi letterbox.")
    print(f"\n  da ghi {dich}")
    print(f"  sha256 {P.bam_file(dich)[:16]}...")
    return 0


# --------------------------------------------------------------------- splits
def cmd_splits(args) -> int:
    _tieu_de("CHIA 5 FOLD THEO CAY - dong bang")
    if P.FOLDS_JSON.exists() and not args.ghi_de:
        print(f"  {P.FOLDS_JSON} DA TON TAI.")
        print("  Sinh lai = moi ket qua train truoc do khong con so sanh duoc voi sau do.")
        print("  Neu that su muon: them --ghi-de")
        return 1

    df_index = bidx.doc_index()
    _, meta = bsplit.tao_split(df_index, index_sha256=P.bam_file(P.INDEX_CSV))
    dich = bsplit.ghi_split(meta)

    print(f"  {meta['n_images']} anh - {meta['n_plants']} cay - {meta['n_leaves']} la")
    print(f"  thuat toan: {meta['thuat_toan']} seed={meta['seed']}")
    print("  khong ro ri o ca muc CAY lan muc LA (da assert)")

    cot = [f"fold{i}" for i in range(meta["n_folds"])]
    print("\n-- so anh moi lop khi fold do lam TEST --")
    bang = pd.DataFrame(meta["so_anh_moi_lop_theo_fold"], index=cot).T
    bang["TONG"] = bang.sum(axis=1)
    print(bang.to_string())
    print("\n-- theo 5 dai bao cao --")
    bb = pd.DataFrame(meta["so_anh_moi_dai_theo_fold"], index=cot).T
    bb["TONG"] = bb.sum(axis=1)
    print(bb.to_string())

    for c in meta["canh_bao"]:
        print(f"\n  [CANH BAO] {c}")
    print(f"\n  da ghi {dich}")
    print("  -> COMMIT file nay vao git ngay. No la nguon su that duy nhat ve cach chia.")
    return 0



# --------------------------------------------------------------------- images
def cmd_images(args) -> int:
    _tieu_de(f"TIEN XU LY ANH - che do '{args.che_do}'")
    df_index = bidx.doc_index()
    kq = bprep.xu_ly(
        df_index,
        che_do=args.che_do,
        size=args.size,
        scale=args.scale,
        neo=args.neo,
        workers=args.workers,
    )
    m = kq.manifest
    print(f"  -> {kq.dich}")
    print(f"  ghi moi {m['n_ghi_moi']} - bo qua (da co) {m['n_bo_qua_da_co']} - {m['giay']}s")
    if m["ti_le_pixel_la"]:
        t = m["ti_le_pixel_la"]
        print(
            f"  ti le pixel LA model thuc nhan: p25 {t['p25'] * 100:.1f}%  "
            f"trung vi {t['trung_vi'] * 100:.1f}%  p75 {t['p75'] * 100:.1f}%"
        )
    return 0


# -------------------------------------------------------------------- package
def cmd_package(args) -> int:
    _tieu_de("DONG GOI DE MANG LEN COLAB")
    if not P.FOLDS_JSON.exists():
        print("  Chua co file split. Chay `splits` truoc.")
        return 1
    thu_muc = P.PROCESSED / bprep.ten_thu_muc(args.che_do, args.size, args.scale, args.neo)
    if not thu_muc.exists():
        print(f"  Chua co anh da xu ly: {thu_muc}. Chay `images` truoc.")
        return 1

    # Giu NGUYEN cay thu muc cua repo ben trong zip. Nho vay tren Colab chi can
    # giai nen + dat COFFEEGUARD_ROOT la moi duong dan trong code chay dung y
    # nguyen — khong sinh ra mot phien ban duong dan thu hai danh rieng cho Colab.
    goi = P.DATA / f"coffeeguard_bench_{thu_muc.name}.zip"
    pngs = sorted(thu_muc.glob("*.png"))
    src_cg = P.ROOT / "src" / "coffeeguard"
    ma_nguon = [src_cg / "__init__.py", *sorted((src_cg / "bench").glob("*.py"))]
    with zipfile.ZipFile(goi, "w", zipfile.ZIP_DEFLATED, compresslevel=6) as z:
        for p in pngs:
            z.write(p, f"data/processed/{thu_muc.name}/{p.name}")
        z.write(thu_muc / "manifest.json", f"data/processed/{thu_muc.name}/manifest.json")
        z.write(P.INDEX_CSV, "data/interim/rocole_index.csv")
        z.write(P.FOLDS_JSON, "data/splits/rocole_folds.json")
        # file chia cua thi nghiem ro ri (neu co) — notebook 02 chi doc khi
        # CACH_CHIA khac "theo_cay", va doc_split chan neu khong khai bao ro.
        for cach in bsplit.CACH_CHIA_RO_RI:
            f = bsplit.duong_dan_split(cach)
            if f.exists():
                z.write(f, f"data/splits/{f.name}")
        # Mang theo chinh ma nguon hop dong: notebook Colab KHONG duoc dinh nghia
        # lai CLASSES_6 hay cach doc split. Hai ban dinh nghia = mot ban se lech,
        # va khi do nhan giai ma ra SAI nhung confidence VAN CAO.
        for p in ma_nguon:
            z.write(p, "src/" + str(p.relative_to(P.ROOT / "src")).replace("\\", "/"))
    mb = goi.stat().st_size / 1048576
    h = P.bam_file(goi)
    (P.DATA / (goi.name + ".sha256")).write_text(f"{h}  {goi.name}\n", encoding="utf-8")
    print(f"  {len(pngs)} PNG + index + split + {len(ma_nguon)} file ma nguon")
    print(f"  -> {goi}  ({mb:.1f} MB)")
    print(f"  sha256 {h}")
    print("\n  Tai len Google Drive: MyDrive/CoffeeGuard/")
    print("  Notebook 02 se KIEM CHUNG sha256 nay truoc khi train.")
    return 0


# ------------------------------------------------------------------- validate
def cmd_validate(_args) -> int:
    _tieu_de("KIEM CHUNG FILE DU DOAN")
    files = bpred.liet_ke()
    if not files:
        print(f"  Khong co file nao trong {P.PREDICTIONS}")
        print("  Notebook train phai xuat bang `predictions.ghi_du_doan(...)`.")
        return 1
    xau = 0
    for f in files:
        try:
            _, cb = bpred.doc_du_doan(f)
            print(f"  [OK] {f.name}")
            for c in cb:
                print(f"       canh bao: {c}")
        except bpred.LoiHopDong as e:
            xau += 1
            print(f"  [LOI] {f.name}\n       {e}")
    print(f"\n  {len(files) - xau}/{len(files)} file hop le.")
    return 1 if xau else 0


# ---------------------------------------- khong-chac / hieu-chinh / kiem-dinh-bai

LUOT_MAC_DINH = ("20261022-1539-luotA", "20260930-1450")


def _doc_theo_luot(luot) -> dict[str, pd.DataFrame]:
    """{arch@run_id: DataFrame} cho các file dự đoán thuộc các lượt `luot`."""
    ra = {}
    for f in bpred.liet_ke():
        _, fw, arch, run = f.stem.split("__")
        if run not in luot:
            continue
        try:
            df, _ = bpred.doc_du_doan(f)
        except bpred.LoiHopDong as e:
            print(f"  [BO QUA] {f.name}: {e}")
            continue
        ra[f"{arch}@{run}"] = df
    return ra


def cmd_khong_chac(args) -> int:
    import itertools

    from coffeeguard.bench import conformal as bcf
    from coffeeguard.bench import tu_choi as btc

    _tieu_de("G2 + G9: LECH KHI BAO TUNG DAI, TAP CONFORMAL THEO DAI")
    preds = _doc_theo_luot(args.luot)
    if not preds:
        print(f"  Khong co file du doan nao cua luot {args.luot}")
        return 1

    lech = []
    for k, d in preds.items():
        for dn in btc.DINH_NGHIA_DUNG:
            b = btc.lech_theo_du_doan(d, n_bootstrap=args.bootstrap, dinh_nghia=dn)
            b.insert(0, "model", k)
            lech.append(b)
    lech = pd.concat(lech, ignore_index=True)
    print("\n-- lech khi bao 'gi sat nang' (duong = qua tu tin) --")
    print(lech[lech.dai_du_doan == "rust_nang"].to_string(index=False))

    # §6.3: hai bien phap hien nhien (ket qua am) — hoc tren fold khac
    nhiet, nguong = [], []
    for k, d in preds.items():
        r = btc.danh_gia_nhiet_do(d, n_bootstrap=args.bootstrap)
        nhiet.append({"model": k, **r})
        b = btc.so_nguong_cung_do_phu(d)
        b.insert(0, "model", k)
        nguong.append(b)
    nhiet, nguong = pd.DataFrame(nhiet), pd.concat(nguong, ignore_index=True)
    print("\n-- temperature scaling cheo fold: ECE va lech khi bao nang (theo dai) --")
    print(nhiet[["model", "T_theo_fold", "ece_truoc", "ece_sau",
                 "lech_dai__rust_nang__truoc", "lech_dai__rust_nang__sau"]].to_string(index=False))
    print("\n-- nguong rieng theo dai vs nguong chung, CUNG do phu --")
    print(nguong.to_string(index=False))

    tom = []
    for k, d in preds.items():
        for a in args.alpha:
            tom.append({"model": k, **bcf.tom_tat(d, alpha=a, so_lan=args.so_lan,
                                                  n_bootstrap=args.bootstrap)})
    tom = pd.DataFrame(tom)
    print("\n-- tap conformal Mondrian, hieu chinh TRONG fold (nua cay), "
          f"trung binh {args.so_lan} lan --")
    print(tom.to_string(index=False))

    cap = []
    for a in args.alpha:
        for x, y in itertools.combinations(preds, 2):
            cap.append(bcf.so_sanh_co_tap(x, y, preds[x], preds[y], alpha=a,
                                          so_lan=args.so_lan, n_bootstrap=args.bootstrap))
    cap = pd.DataFrame(cap)
    print("\n-- hieu co tap TB (a - b, am = a hep hon), CI95 theo cay --")
    print(cap.to_string(index=False))

    P.REPORTS.mkdir(parents=True, exist_ok=True)
    ten_file = ("lech_theo_dai", "nhiet_do", "nguong_theo_dai", "conformal_theo_dai",
                "conformal_cap")
    for ten, bang in zip(ten_file, (lech, nhiet, nguong, tom, cap)):
        bang.to_csv(P.REPORTS / f"{ten}.csv", index=False, encoding="utf-8")
    print(f"\n  da ghi {P.REPORTS}/{{{', '.join(ten_file)}}}.csv")
    return 0


def _doc_cap_do_theo_cay() -> dict[str, pd.DataFrame]:
    """{arch@run_id: DataFrame} cho ResNet18/50/152 chia theo cay (notebook 02b_cap_do_*)."""
    ra = {}
    for f in sorted(P.PREDICTIONS_CAP_DO.glob("pred__*-theo_cay.csv")):
        _, fw, arch, run = f.stem.split("__")
        ra[f"{arch}@{run}"] = pd.read_csv(f)
    return ra


def cmd_hieu_chinh(args) -> int:
    """Phan bien ban nhap G2+G9: moi cach hieu chinh tren CUNG so do chia, them doi chung."""
    from coffeeguard.bench import conformal as bcf
    from coffeeguard.bench import hieu_chinh as bhc
    from coffeeguard.bench import metrics as bm
    from coffeeguard.bench import tu_choi as btc

    _tieu_de("G2 + G9: CUNG SO DO HIEU CHINH, DOI CHUNG VECTOR/DIRICHLET, APS/RAPS/THU BAC")
    preds = _doc_cap_do_theo_cay() if args.cap_do else _doc_theo_luot(args.luot)
    hau_to = "__cap_do" if args.cap_do else ""
    if not preds:
        print("  Khong co file du doan nao")
        return 1

    # Bang 2 cua bai: do phu o nguong co dinh, tach theo dai that
    phu = []
    for k, d in preds.items():
        b = btc.duong_do_phu(d, nguong=(args.nguong,), n_bootstrap=args.bootstrap)
        yt = [ct.band_of(c) for c in d["y_true"]]
        b.insert(0, "macro_f1_dai_toan_bo", round(bm.macro_f1(yt, [ct.band_of(c) for c in d["y_pred"]],
                                                      ct.CLASSES_BAND), 4))
        b.insert(0, "model", k)
        phu.append(b)
    phu = pd.concat(phu, ignore_index=True)
    print(f"\n-- do phu o nguong {args.nguong} theo dai that --")
    print(phu.to_string(index=False))

    hc = pd.DataFrame([{"model": k, **bhc.danh_gia(d, pp, h, so_lan=args.so_lan,
                                                    n_bootstrap=args.bootstrap)}
                       for k, d in preds.items() for pp in bhc.PHUONG_PHAP
                       for h in bcf.HIEU_CHINH])
    print("\n-- hieu chinh xac suat: ECE va lech khi bao nang, hai so do --")
    print(hc[["model", "phuong_phap", "hieu_chinh", "ece_truoc", "ece_sau", "macro_f1_dai_sau",
              "n_bao_nang_sau", "lech_nang_truoc", "lech_nang_sau", "lech_nang_sau_ci_lo",
              "lech_nang_sau_ci_hi"]].to_string(index=False))

    ng = []
    for k, d in preds.items():
        for h in bcf.HIEU_CHINH:
            for n in (10, 5):
                b = bhc.nguong_theo_dai(d, h, n_toi_thieu=n, so_lan=args.so_lan)
                b.insert(0, "model", k)
                ng.append(b)
    ng = pd.concat(ng, ignore_index=True)
    print("\n-- nguong theo dai vs nguong chung, cung do phu, hai so do --")
    print(ng.to_string(index=False))

    cp = pd.DataFrame([{"model": k, **bcf.tom_tat(d, alpha=a, so_lan=args.so_lan,
                                                   n_bootstrap=args.bootstrap, diem=di,
                                                   che_do=cd, hieu_chinh=h)}
                       for k, d in preds.items() for a in args.alpha for di in bcf.CACH_DIEM
                       for cd in bcf.CHE_DO for h in bcf.HIEU_CHINH])
    print("\n-- conformal: diem x che do x so do hieu chinh --")
    print(cp[["model", "alpha", "diem", "che_do", "hieu_chinh", "co_tap_tb", "ty_le_rong",
              "ty_le_lien_nhau", "ty_le_nguong_vo_cuc"]
             + [f"phu__{b}" for b in ct.CLASSES_BAND]].to_string(index=False))

    P.REPORTS.mkdir(parents=True, exist_ok=True)
    ten_file = ("phu_nguong_co_dinh", "hieu_chinh_cung_so_do", "nguong_cung_so_do",
                "conformal_doi_chung")
    for ten, bang in zip(ten_file, (phu, hc, ng, cp)):
        bang.to_csv(P.REPORTS / f"{ten}{hau_to}.csv", index=False, encoding="utf-8")
    print(f"\n  da ghi {P.REPORTS}/{{{', '.join(ten_file)}}}{hau_to}.csv")
    return 0


def cmd_kiem_dinh_bai(args) -> int:
    """Phan bien bai G2+G9 (ban GV): Holm cho cac cap co tap cung luot, nguong +inf theo dai."""
    import itertools

    from coffeeguard.bench import conformal as bcf
    from coffeeguard.bench import contract as ct
    from coffeeguard.bench import metrics as bm

    _tieu_de("G2 + G9: HOLM CHO CAP CO TAP CUNG LUOT, NGUONG +INF THEO DAI")
    preds = _doc_theo_luot(args.luot)
    if not preds:
        print("  Khong co file du doan nao")
        return 1

    # Chi cap CUNG luot (khong ghep hai lan huan luyen khac nhau) — nhu Bang 7 cua bai
    cap = []
    for x, y in itertools.combinations(sorted(preds), 2):
        if x.split("@")[1] != y.split("@")[1]:
            continue
        cap.append(bcf.so_sanh_co_tap(x, y, preds[x], preds[y], alpha=args.alpha_cap,
                                      so_lan=args.so_lan, n_bootstrap=args.bootstrap))
    cap = pd.DataFrame(cap)
    cap["p_holm"] = bm.holm(cap["p_bootstrap"])
    cap["co_y_nghia_holm"] = cap["p_holm"] <= 0.05
    print(f"\n-- {len(cap)} cap cung luot, alpha = {args.alpha_cap}: p bootstrap va Holm --")
    print(cap[["a", "b", "hieu", "ci_lo", "ci_hi", "p_bootstrap", "p_holm",
               "co_y_nghia_holm"]].to_string(index=False))
    print(f"  con y nghia sau Holm (p_holm <= 0,05): {int(cap.co_y_nghia_holm.sum())}/{len(cap)}")

    vc = []
    for k, d in preds.items():
        for a in args.alpha:
            _, tt = bcf.tap_du_doan(d, alpha=a, so_lan=args.so_lan)
            so_hieu_chinh = tt["phu_nua"].shape[0] * tt["phu_nua"].shape[1]
            dong = {"model": k, "alpha": a, "so_lan_hieu_chinh_moi_dai": so_hieu_chinh,
                    "tong_vo_cuc": int(tt["vo_cuc_dai"].sum()),
                    "ty_le_nguong_vo_cuc": tt["ty_le_nguong_vo_cuc"]}
            for c, b in enumerate(ct.CLASSES_BAND):
                dong[f"vo_cuc__{b}"] = int(tt["vo_cuc_dai"][c])
            vc.append(dong)
    vc = pd.DataFrame(vc)
    print("\n-- so nguong +inf theo dai (conformal theo dai, 1 - p, nua cay) --")
    print(vc.to_string(index=False))

    P.REPORTS.mkdir(parents=True, exist_ok=True)
    cap.to_csv(P.REPORTS / "conformal_cap_holm.csv", index=False, encoding="utf-8")
    vc.to_csv(P.REPORTS / "conformal_vo_cuc_theo_dai.csv", index=False, encoding="utf-8")
    print(f"\n  da ghi {P.REPORTS}/{{conformal_cap_holm, conformal_vo_cuc_theo_dai}}.csv")
    return 0



def main() -> int:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    sp = ap.add_subparsers(dest="cmd", required=True)

    sp.add_parser("doctor", help="kiem tra du lieu & thu vien").set_defaults(fn=cmd_doctor)
    sp.add_parser("index", help="dung bang chi muc anh").set_defaults(fn=cmd_index)

    p = sp.add_parser("splits", help="chia 5 fold theo cay (dong bang)")
    p.add_argument("--ghi-de", action="store_true", help="BAT BUOC neu muon sinh lai")
    p.set_defaults(fn=cmd_splits)

    for ten, fn in (("images", cmd_images), ("package", cmd_package)):
        p = sp.add_parser(ten)
        p.add_argument("--che-do", default="leafsquare", choices=list(ct.CHE_DO_CAT))
        p.add_argument("--size", type=int, default=ct.IMG_SIZE)
        p.add_argument("--scale", type=float, default=ct.LEAF_SCALE)
        p.add_argument(
            "--neo", default=ct.LEAF_NEO, choices=["ngan", "dai"],
            help="neo canh o vuong vao canh NGAN hay DAI cua khung bao la",
        )
        if ten == "images":
            p.add_argument("--workers", type=int, default=8)
        p.set_defaults(fn=fn)

    p = sp.add_parser("khong-chac", help="G2+G9: lech theo dai, tap conformal theo dai")
    p.add_argument("--luot", nargs="+", default=list(LUOT_MAC_DINH))
    p.add_argument("--alpha", nargs="+", type=float, default=[0.10, 0.20])
    p.add_argument("--so-lan", type=int, default=20, help="so lan chia nua cay de hieu chinh")
    p.add_argument("--bootstrap", type=int, default=2000)
    p.set_defaults(fn=cmd_khong_chac)
    p = sp.add_parser("hieu-chinh",
                      help="G2+G9: cung so do hieu chinh, vector/Dirichlet, APS/RAPS/thu bac")
    p.add_argument("--luot", nargs="+", default=list(LUOT_MAC_DINH))
    p.add_argument("--cap-do", action="store_true",
                   help="ResNet18/50/152 chia theo cay trong reports/predictions_cap_do/")
    p.add_argument("--alpha", nargs="+", type=float, default=[0.10, 0.20])
    p.add_argument("--nguong", type=float, default=0.7, help="nguong tin cay chung (Bang 2)")
    p.add_argument("--so-lan", type=int, default=20, help="so lan chia nua cay de hieu chinh")
    p.add_argument("--bootstrap", type=int, default=2000)
    p.set_defaults(fn=cmd_hieu_chinh)
    p = sp.add_parser("kiem-dinh-bai",
                      help="G2+G9: Holm cho cap co tap cung luot, nguong +inf theo dai")
    p.add_argument("--luot", nargs="+", default=list(LUOT_MAC_DINH))
    p.add_argument("--alpha", nargs="+", type=float, default=[0.10, 0.20])
    p.add_argument("--alpha-cap", type=float, default=0.20, help="alpha cua phep so co tap")
    p.add_argument("--so-lan", type=int, default=20)
    p.add_argument("--bootstrap", type=int, default=2000)
    p.set_defaults(fn=cmd_kiem_dinh_bai)

    sp.add_parser("validate", help="kiem chung file du doan").set_defaults(fn=cmd_validate)

    a = ap.parse_args()
    return a.fn(a)


if __name__ == "__main__":
    raise SystemExit(main())
