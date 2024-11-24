# GAPY - Genetic Algorithm for the Traveling Salesman Problem

最佳旅行者問題解決方案

## 專案簡介

這個專案使用基因演算法來解決著名的最佳旅行者問題（Traveling Salesman Problem, TSP）。最佳旅行者問題是一個組合優化問題，目的是尋找一條最短的路徑，使旅行者能夠訪問每個城市一次並返回起點。其中還使用了 multiprocessing 來加速運算。

### 最佳旅行者問題

最佳旅行者問題是一個組合優化問題，目的是尋找一條最短的路徑，使旅行者能夠訪問每個城市一次並返回起點。這個問題是一個 NP-hard 問題，因此無法在多項式時間內找到最佳解。基因演算法是一種啟發式算法，通常用於解決這種組合優化問題。

### 基因演算法

基因演算法是一種模仿自然選擇和遺傳機制的優化算法。在基因演算法中，每個解都表示為一個染色體，染色體由一系列基因組成。基因演算法通過模擬自然選擇和遺傳機制來搜索最佳解。在每一代中，基因演算法會根據染色體的適應度對解進行排序，然後根據適應度選擇父代進行交叉和變異，生成下一代。通過重複這個過程，基因演算法可以找到一個接近最佳解的解。

#### 染色體表示

在這個專案中，我們定義染色體表示為城市的順序。例如，染色體 `[0, 1, 2, 3]` 表示旅行者依次訪問城市 0、1、2 和 3。

#### 適應度函數

在演化過程中，我們會計算每個個體的適應度，若他們的路徑長度愈短，則分數愈高。

我們定義適應度函數為

$$
fitness(i) = \frac{n - rank(i)}{distance(i)}
$$

其中 $n$ 為個體數量，$rank(i)$ 為個體 $i$ 的排名，$distance(i)$ 為個體 $i$ 的路徑長度。

#### 交配

首先，我們只會保留適應度較高的前幾名個體，然後根據適應度進行擇偶。我們使用輪盤選擇法來選擇兩個父代來交配，直到生成下一代的個體數量達到原來的個體數量。

其中，輪盤選擇法選中個體 $i$ 的機率為

$$
P(i) = \frac{fitness(i)}{\sum_{j=1}^{n} fitness(j)}
$$

然後，我們會將選中的兩個父代 (F1, F2) 進行交配，生成下一代的個體，其步驟如下。

1.  將 F1 切成 $k$ 段，並取出第 $0$、$2$、$4$、$\ldots$、$k$ 段。
    例如：F1 為 `0 1 2 3 4 5 6 7`，並切成 $4$ 段，
    則取出 `0 1`、`4 5`。
2.  將 F2 中未被 F1 選擇的基因取出，並切成 $k/2$ 段。
    例如：F2 為 `0 7 2 6 5 3 4 1`，
    則 F2 只能選擇 `7 2 6 3`，並切成 `7 2`、`6 3`。
3.  將 F1 與 F2 的基因交叉，生成下一代的個體。
    例如：F1 為 `0 1`、`4 5`，F2 為 `7 2`、`6 3`，
    則下一代的個體為 `0 1`、`7 2`、`4 5`、`6 3`。
4.  對於每一個基因，皆有一定概率進行突變。
    其突變方式為將兩個基因進行交換。
5.  重複步驟 1-4，直到生成下一代的個體數量達到原來的個體數量。

## 安裝

請確保已安裝 Python 3.12 或更高版本，然後執行以下命令來安裝所需的依賴項：

```bash
pip install -r requirements.txt
```

## 使用方法

1.  編輯配置文件 `config.yaml` 以設置參數

2.  執行主程序：

```bash
python main.py
```

3.  程序運行完畢後，
    會在 `results` 目錄下生成各代最優解的路徑的圖片，
    以及 `history.json` 文件紀錄每一代的概況。

## 文件結構

-   `main.py`：主程序文件
-   `config.yaml`：配置文件
-   `display.py`：包含結果可視化的函數
-   `ga.py`：包含基因演算法的實現
-   `location.py`：包含城市和旅行者的定義
-   `requirements.txt`：依賴項列表
-   `utility.py`：包含輔助函數
