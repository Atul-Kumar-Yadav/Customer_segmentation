from statistics import mode
from mpl_toolkits.mplot3d import Axes3D
import seaborn as sns 
from sklearn.preprocessing import MinMaxScaler
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.neighbors import NearestNeighbors
import streamlit as st
from scipy.cluster.hierarchy import linkage, dendrogram
from streamlit.components.v1 import html
import re
import io
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans, DBSCAN, AgglomerativeClustering, MiniBatchKMeans
from sklearn.metrics import silhouette_score
from sklearn.decomposition import PCA

# setting up page configuration
st.set_page_config(layout="wide")

#setting up cache method for loading data
@st.cache_data
def load_file(file) -> pd.DataFrame:
    file_name = file.name.lower()
    if file_name.endswith('.csv') :
        encodings = ["utf-8", "utf-8-sig", "latin-1"]
        for enc in encodings:
            try:
                df = pd.read_csv(file, encoding=enc, low_memory=False)
                return df
            except:
                continue
            raise ValueError("csv  encoding is not supported.")
    elif file_name.endswith(('.xlsx', '.xls')):
        df = pd.read_excel(file)
        return df
    else:
        raise ValueError("File type is not supported.")
label_sets = {
    3: ["High", "Medium", "Low"],
    4: ["Very High", "High", "Low", "Very Low"],
    5: ["Extremely High", "High", "Average", "Low", "Extremely Low"],
    6: ["Extremely High", "Very High", "High", "Low", "Very Low", "Extremely Low"],
    7: ["Extremely High", "Very High", "High", "Average", "Low", "Very Low", "Extremely Low"],
    8: ["Extremely High", "Very High", "High", "Above Average", "Below Average", "Low", "Very Low", "Extremely Low"],
    9: ["Extremely High", "Very High", "High", "Above Average", "Average", "Below Average", "Low", "Very Low", "Extremely Low"],
}
def build_segment_map(x_df, labels, lower_is_better=None):
    lower_is_better = set(lower_is_better or [])
    tmp = x_df.copy()
    tmp["cluster"] = labels

    profile = tmp.groupby("cluster").mean(numeric_only=True)

    scaled = pd.DataFrame(
        MinMaxScaler().fit_transform(profile),
        index=profile.index,
        columns=profile.columns
    )

    score = 0
    for c in scaled.columns:
        score += (1 - scaled[c]) if c in lower_is_better else scaled[c]

    ordered = score.sort_values(ascending=False).index.tolist()
    names = label_sets[len(ordered)]
    cmap = {cid: names[i] for i, cid in enumerate(ordered)}

    return cmap
# Function to clean column names by removing special characters and extra spaces
def clean_column_names(df):

    original_cols = df.columns.tolist()

    cleaned_cols = [
        re.sub(r'[^a-z0-9]', "", col.lower())
        for col in df.columns
    ]
    df.columns = cleaned_cols

    clean_original = dict(zip(cleaned_cols, original_cols))
    original_clean = dict(zip(original_cols, cleaned_cols))
    
    return df, clean_original, original_clean
  #detecting rfm applicable or not   
def check_rfm(df):
    customer_columns = []
    date_columns = []
    monetary_columns = []

    customer_key = ['customer', 'custid', 'customerid', 'clientid', 'client','id']
    for col in df.columns:
        if any(key in col for key in customer_key):
            customer_columns.append(col)
            
        date_key = ['invoicedate','orderdate','date', 'time', 'timestamp', 'invoice', 'order', 'purchase','recency', 'frequency']
    for col in df.columns:
        if any(key in col for key in date_key):
            date_columns.append(col)
            
    monetary_key = ['unitprice', 'quantity', 'price','unit','monetary','amount', 'cost', 'revenue', 'sales','total',]
    for col in df.columns:
        if any(key in col for key in monetary_key):
            monetary_columns.append(col)
            
    
    rfm_applicable= False
    if customer_columns and date_columns and monetary_columns:
        rfm_applicable = True
    return {
        'customer_col' : customer_columns,
        'date_col' : date_columns,
        'monetary_col' : monetary_columns,
        'rfm_applicable': rfm_applicable
    }


def handle_outliers(data: pd.DataFrame) -> pd.DataFrame:
    df = data.copy()

    # only numeric columns from your selected features
    num = df.select_dtypes(include="number")
    if num.empty:
        return df

    q1 = num.quantile(0.25)
    q3 = num.quantile(0.75)
    iqr = q3 - q1

    lower = q1 - 1.5 * iqr
    upper = q3 + 1.5 * iqr

    # row is outlier if any selected numeric feature is outside bounds
    outlier_rows = ((num < lower) | (num > upper)).any(axis=1)

    return df.loc[~outlier_rows]


def numeric_columns(df: pd.DataFrame):
    return df.select_dtypes(include=[np.number]).columns.tolist()

def detect_basic_columns(df):

    income_keywords = ["income", "annualincome"]
    spending_keywords = ["spending", "spendingscore", "score"]

    income_cols = []
    spending_cols = []

    for col in df.columns:

        if any(key in col for key in income_keywords):
            income_cols.append(col)

        if any(key in col for key in spending_keywords):
            spending_cols.append(col)

    basic_clustering = bool(income_cols and spending_cols)

    return {
        "income_cols": income_cols,
        "spending_cols": spending_cols,
        "basic_clustering": basic_clustering
    }


            
def auto_best_clustering(data,algorithm : None):

    results = {}
    algo = algorithm

    # 🔹 1. Scale data
    scaler = StandardScaler()
    data_scaled = scaler.fit_transform(data)

    # 🔹 2. Reduce dimension (for speed + stability)
    if data_scaled.shape[1] > 3:
        pca = PCA(n_components=min(3, data_scaled.shape[1]))
        data_reduced = pca.fit_transform(data_scaled)
    else:
        data_reduced = data_scaled

    n = len(data)
    
    if algo is not None:
        algos = [algo]
    else:
        # 🔹 3. Select algorithms based on size
        if n < 1000:
            algos = ["kmeans", "dbscan", "hierarchical"]
        elif n < 4000:
            algos = ["kmeans", "dbscan"]
        elif n < 20000:
            algos = ["kmeans", "minibatch_kmeans"]
        else:
            algos = ["minibatch_kmeans"]

    # ---------------------------
    # 🔹 KMEANS
    # ---------------------------
    if "kmeans" in algos:
        best_score = -1
        best_labels = None
        cluster_size = None
        k_inertia = []

        for k in range(3, 10):
            model = KMeans(n_clusters=k, n_init=10)
            labels = model.fit_predict(data_reduced)
            k_inertia.append(model.inertia_)

            if len(set(labels)) > 1:
                score = silhouette_score(data_reduced, labels)

                if score > best_score:
                    best_score = score
                    best_labels = labels
                    cluster_size = k

        if best_labels is not None:
            results["kmeans"] = (best_score, best_labels, cluster_size, k_inertia)

    # ---------------------------
    # 🔹 MINIBATCH KMEANS
    # ---------------------------
    if "minibatch_kmeans" in algos:
        best_score = -1
        best_labels = None
        cluster_size = None
        k_inertia = []

        for k in range(3, 10):
            model = MiniBatchKMeans(n_clusters=k, batch_size=256)
            labels = model.fit_predict(data_reduced)
            k_inertia.append(model.inertia_)

            if len(set(labels)) > 1:
                score = silhouette_score(data_reduced, labels)

                if score > best_score:
                    best_score = score
                    best_labels = labels
                    cluster_size = k

        if best_labels is not None:
            results["minibatch_kmeans"] = (best_score, best_labels, cluster_size, k_inertia)

    # ---------------------------
    # 🔹 DBSCAN
    # ---------------------------
    if "dbscan" in algos:
        best_score = -1
        best_labels = None
        cluster_size = None
        coarse_range = np.linspace(0.1, 1.0, num=15)
        cluster_size = None
        k_inertia = None

        for eps in coarse_range:
            model = DBSCAN(eps=eps, min_samples=10)
            labels = model.fit_predict(data_reduced)
            n_clusters = len(set(labels)) - (1 if -1 in labels else 0)

            if 3 <= n_clusters <= 9:
                try:
                    score = silhouette_score(data_reduced, labels)

                    if score > best_score:
                        best_score = score
                        best_labels = labels
                        cluster_size = n_clusters
                except:
                    continue 


        if best_labels is not None:
            results["dbscan"] = (best_score, best_labels, cluster_size, k_inertia)

    # ---------------------------
    # 🔹 HIERARCHICAL
    # ---------------------------
    if "hierarchical" in algos:
        best_score = -1
        best_labels = None
        cluster_size = None
        k_inertia = None    

        for k in range(3, 10):
            model = AgglomerativeClustering(n_clusters=k)
            labels = model.fit_predict(data_reduced)

            if len(set(labels)) > 1:
                score = silhouette_score(data_reduced, labels)

                if score > best_score:
                    best_score = score
                    best_labels = labels
                    cluster_size = k

        if best_labels is not None:
            results["hierarchical"] = (best_score, best_labels, cluster_size, k_inertia)

    # ---------------------------
    # 🔹 SELECT BEST
    # ---------------------------
    if not results:
        return None, None, None, None, None

    best_algo = max(results, key=lambda x: results[x][0])
    best_score, best_labels, cluster_size, k_inertia = results[best_algo]

    return best_algo, best_score, best_labels, cluster_size, k_inertia 
           
def plot_cluster( 
        x,
        mode,                 # "Basic_Mode" | "RFM_Mode" | "Manual_Mode"
        algorithm,            # feature data in numerical format
        score,                 #silhouette score
        factor2,               # inertia /eps
        labels,               # cluster labels
        cluster_no,           # selected clusters (for kmeans-like) 
        features,
        segment_map=None         # mapping of cluster label to segment name (optional
):
    st.markdown( f"**Algorithm:** {algorithm} / **Score:** {score} / **Number of Clusters:** {cluster_no} " )
    Left,Right = st.columns(2)
    with Left:
        if algorithm in ["kmeans", "minibatch_kmeans"]:
            elbow_k = list(range(3, 10)) 
            # Plot elbow curve
            fig_l, ax_l = plt.subplots(figsize=(7, 5))
            ax_l.plot(elbow_k, factor2, marker="o")
            ax_l.axvline(cluster_no, color="red", linestyle="--", label=f"k={cluster_no}")
            ax_l.set_title("Elbow Curve")
            ax_l.set_xlabel("Number of Clusters (k)")
            ax_l.set_ylabel("Inertia")
            ax_l.legend()

        elif algorithm == "dbscan":
            # plot k-distance graph
            nn = NearestNeighbors(n_neighbors=10)
            nn.fit(x)
            distances, _ = nn.kneighbors(x)
            k_dist = np.sort(distances[:, -1])  # distance to 10th
            fig_l, ax_l = plt.subplots(figsize=(7, 5))
            ax_l.plot(k_dist)
            ax_l.set_title(f"DBSCAN k-distance Plot (k=10)")
            ax_l.set_xlabel("Points sorted by distance")   
            ax_l.set_ylabel("Distance to 10th Nearest Neighbor")

        elif algorithm == "hierarchical":
            # plot dendogram
            linkage_matrix = linkage(x, method="ward")
            fig_l, ax_l = plt.subplots(figsize=(7, 5))
            dendrogram(linkage_matrix, ax=ax_l, no_labels=True)
            ax_l.set_title("Hierarchical Clustering Dendrogram")
            ax_l.set_xlabel("Samples")
            ax_l.set_ylabel("Distance")

        else:
            ax_l.text(0.5, 0.5, "No specific plot for this algorithm", ha="center", va="center")
            ax_l.axis("off")

        st.pyplot(fig_l, use_container_width=True)
        plt.close(fig_l) 
        
    with Right:
        if mode == "Basic_Mode" or mode == "Manual_Mode" and x.shape[1] == 2:
            fig_r,ax_r = plt.subplots(figsize=(7,5))
            if x.shape[1]<2:
                ax_r.text(0.5, 0.5, "Not enough features to plot clusters", ha="center", va="center")
                ax_r.axis("off")
            else:
                unique_clusters = sorted(np.unique(labels))
                cmap = plt.cm.get_cmap("tab10", len(unique_clusters))

                for i, cid in enumerate(unique_clusters):
                    m = labels == cid
                    seg_name = segment_map.get(cid, "Unknown")  # segment name only
                    ax_r.scatter(
                        x[m, 0], x[m, 1],
                        color=cmap(i),
                        alpha=0.85,
                        s=30,
                        label=seg_name
                    )
                ax_r.set_title("Cluster Visualization 2D", fontsize=12,pad=10)  
                ax_r.set_xlabel(features[0])    
                ax_r.set_ylabel(features[1])
                ax_r.legend(title="customer importance")

        elif mode == "RFM_Mode" or mode == "Manual_Mode" and x.shape[1] >= 3:
            fig_r, ax_r = plt.subplots(figsize=(5, 5), subplot_kw={"projection": "3d"})

            if x.shape[1] < 3:
                ax_r.text(0.5, 0.5, "Need 3 features to plot clusters", ha="center", va="center")
                ax_r.axis("off")
            else:
                unique_clusters = sorted(np.unique(labels))
                cmap = plt.cm.get_cmap("tab10", len(unique_clusters))

                for i, cid in enumerate(unique_clusters):
                    m = labels == cid
                    seg_name = segment_map.get(cid, "Unknown")
                    ax_r.scatter(
                        x[m, 0], x[m, 1], x[m, 2],
                        color=cmap(i), alpha=0.85, s=30, label=seg_name
                    )
                ax_r.set_title("Cluster Visualization 3D",fontsize=8,pad=0 )  
                ax_r.set_xlabel(features[0],labelpad=1,fontsize=7)    
                ax_r.set_ylabel(features[1],labelpad=1,fontsize=7)
                ax_r.set_zlabel(features[2],labelpad=-17,fontsize=7,rotation=90)
                fig_r.subplots_adjust(left=0.08, right=0.95, bottom=0.16, top=0.90)
                ax_r.tick_params(axis='x', labelsize=8,pad=5)
                ax_r.tick_params(axis='y', labelsize=8,pad=5)
                ax_r.tick_params(axis='z', labelsize=8,pad=5)
                ax_r.legend(title="Customer Importance", fontsize=7, title_fontsize=8, loc="best")



        st.pyplot(fig_r, use_container_width=True)
        plt.close(fig_r)

                


                
        

        
st.markdown(
    """
    <style> 
    .heading h1 {text-align : center; color:purple;margin-bottom:30px;}
    
      .stButton button, .stDownloadButton button { border-radius: 12px; }
      div[data-testid="stMetricValue"] { font-size: 1.6rem; }
      .card {
        border-radius: 18px;
        padding: 14px 16px;
        border: 1px solid rgba(255,255,255,0.10);
        background: rgba(255,255,255,0.04);
      }
      .muted { color: rgba(255,255,255,0.70); }
      .small { font-size: 0.92rem; }
      .stButton button, .stDownloadButton button { border-radius: 12px; }
        div[data-testid="stMetricValue"] { font-size: 1.6rem; }
    </style>
    """,
    unsafe_allow_html=True
)


st.markdown(
    """
    <div class= heading>
    <h1>Customer Segmentation Using Clustering</h1>
    </div>
    """ ,
    unsafe_allow_html =True
)

uploaded = st.file_uploader(" ⬆️ Click below to Upload file" , type=["csv", "xlsx", "xls"])
if not uploaded :
    st.info("⬆️ Upload the CSV or Excel file to begin")
    st.stop()
df=load_file(uploaded)

num_cols = numeric_columns(df)

top = st.columns([2, 1, 1, 1])
with top[0]:
    st.markdown('<div class="card"><div class="small muted">Preview</div></div>', unsafe_allow_html=True)
with top[1]:
    st.metric("Rows", f"{len(df):,}")
with top[2]:
    st.metric("Columns", f"{df.shape[1]:,}")
with top[3]:
    st.metric("Numeric cols", f"{len(num_cols):,}")


st.write(df.head(100))
df, clean_original, original_clean = clean_column_names(df)



mode = None
manual_mode = st.toggle("Manual(skip auto-detection)", value=False, help="Toggle on to select features and algorithm manually.")

if not manual_mode:
    basic_info = detect_basic_columns(df)

    if basic_info["basic_clustering"]:
        st.success("Basic clustering features detected!")
        mode = "Basic_Mode"
        col = "customerid"
        features =[]
        features.extend(basic_info["income_cols"])
        features.extend(basic_info["spending_cols"])
        # remove duplicate features
        features = list(set(features))
        original_features = [clean_original.get(f, f) for f in features]

        st.write(f"Identified features for clustering: {original_features}")

        c1,c2,c3 = st.columns([1,1,1])
        with c1:
            st.markdown('<div class="card"><div class="Small Muted">Income features</div></div>', unsafe_allow_html=True)
            col_x = st.selectbox("X_feature",options= original_features, index=0)
            col_x1 = original_clean.get(col_x, col_x)
        with c2:
            st.markdown('<div class="card"><div class="Small Muted">Spending features</div></div>', unsafe_allow_html=True)
            col_y = st.selectbox("Y_feature",options= original_features, index=1)
            col_y1 = original_clean.get(col_y, col_y)
        with c3:
            st.markdown('<div class="card"><div class="small muted">Income features</div></div><br>', unsafe_allow_html=True)
            st.write("hint")
            
        x1=df[[col_x1,col_y1]]
        x1 = x1[x1 > 0]
        x1= handle_outliers(x1).dropna()

        if x1.empty:    
                st.error("After dropping missing values, there are no rows left. Please clean your CSV or pick other columns.")
                st.stop()
           
        x = x1.values
        

        run = st.button("🚀 Run clustering", type="primary")
        if not run:
            st.stop()   
         
        algo, score, labels, no_cluster, factor2  = auto_best_clustering(x, None)
        
        if algo is None:
            st.error("Clustering failed. Please try manual mode or clean your data.")
            st.stop()
        else:
            st.success(f"Best algorithm: {algo} with silhouette score: {score:.2f}")
            segment_map = build_segment_map(x1, labels)
            df["cluster"] = np.nan
            df.loc[x1.index, "cluster"] = labels
            df["Customer_Importance"] = df["cluster"].map(segment_map)

            st.write(df)
            st.write(f"Number of clusters: {no_cluster}")
            plot_cluster(x, mode, algo, score, factor2, labels, no_cluster, [col_x, col_y], segment_map=segment_map)
            


    else:
        st.warning("Basic features not found. Checking RFM...")

        rfm_info = check_rfm(df)

        if rfm_info['rfm_applicable']:
            st.success("RFM features detected!")
            mode = "RFM_Mode"
            features = []
            features.extend(rfm_info['customer_col'])
            features.extend(rfm_info['date_col'])
            features.extend(rfm_info['monetary_col'])   
            features = list(set(features))
            original_features = [clean_original.get(f, f) for f in features]
            st.write(f"Identified features for RFM clustering: {original_features}")
            st.info(" If Receny , Frequency and Moetary column is given directly then press below")
            rfm_direct = st.toggle("RFM columns given directly", value=False)
            if rfm_direct:
                with st.expander("Select RFM columns"):
                    c1,c2,c3 = st.columns([1,1,1])
                    with c1:
                        st.markdown('<div class="card"><div class="Small Muted">Recency</div></div>', unsafe_allow_html=True)
                        col_r = st.selectbox("Recency", options=original_features, index=0)
                        col_r1 = original_clean.get(col_r, col_r)
                    with c2:
                        st.markdown('<div class="card"><div class="Small Muted">Frequency</div></div>', unsafe_allow_html=True)
                        col_f = st.selectbox("Frequency", options=original_features, index=1)
                        col_f1 = original_clean.get(col_f, col_f)
                    with c3:
                        st.markdown('<div class="card"><div class="Small Muted">Monetary Value</div></div>', unsafe_allow_html=True)
                        col_monetary = st.selectbox("Monetary Value", options=original_features, index=2)
                        col_monetary1 = original_clean.get(col_monetary, col_monetary)
                x1 = df[[col_r1, col_f1, col_monetary1]].dropna()
                x1 = x1[x1 > 0]
                if x1.empty:
                    st.error("After dropping missing values, there are no rows left. Please clean your CSV or pick other columns.")
                    st.stop()
                # x1= handle_outliers(x1).dropna()
                x1 = x1.dropna()

                if x1.empty:
                    st.error("After handling outliers, there are no rows left. Please clean your CSV or pick other columns.")
                    st.stop()
                x = x1.values

                

                    
                    

            else:

                st.info("Select the following features as per their labels shown below")
                c1,c2,c3,c4,c5 = st.columns([1,1,1,1,1])
                with c1:
                        st.markdown('<div class="card"><div class="small muted">Customer ID</div></div>', unsafe_allow_html=True)
                        col_cust = st.selectbox("Customer ID", options=original_features, index=0)
                        col_cust1 = original_clean.get(col_cust, col_cust)
                with c2:
                        st.markdown('<div class="card"><div class="small muted">Invoice no.</div></div>', unsafe_allow_html=True)
                        col_invoice_no = st.selectbox("Invoice no.", options=original_features, index=1)
                        col_invoice_no1 = original_clean.get(col_invoice_no, col_invoice_no)
                with c3:
                        st.markdown('<div class="card"><div class="small muted">Invoice Date</div></div>', unsafe_allow_html=True)
                        col_date = st.selectbox("Invoice Date", options=original_features, index=2)
                        col_date1 = original_clean.get(col_date, col_date)
                with c4:
                        st.markdown('<div class="card"><div class="small muted">Quantity</div></div>', unsafe_allow_html=True)
                        col_quantity = st.selectbox("Quantity", options=original_features, index=3)
                        col_quantity1 = original_clean.get(col_quantity, col_quantity)
                with c5:
                        st.markdown('<div class="card"><div class="small muted">Unit Price</div></div>', unsafe_allow_html=True)
                        col_unit = st.selectbox("Unit Price", options=original_features, index=4)
                        col_unit1 = original_clean.get(col_unit, col_unit)


                x1 = df[[col_cust1, col_invoice_no1, col_date1, col_quantity1, col_unit1]].dropna()
                if x1.empty:
                    st.error("After dropping missing values, there are no rows left. Please clean your CSV or pick other columns.")
                    st.stop()
                st.spinner("Calculating RFM features...")
                
                x1 = x1.copy()
                x1[col_date1] = pd.to_datetime(x1[col_date1], errors='coerce')
                x1[col_quantity1] = pd.to_numeric(x1[col_quantity1], errors='coerce')
                x1[col_unit1] = pd.to_numeric(x1[col_unit1], errors='coerce')
                x1 = x1[(x1[col_quantity1] > 0) & (x1[col_unit1] > 0)]
                x1 = x1.dropna(subset=[col_date1, col_quantity1, col_unit1])
                st.write(x1.shape)
                #monetary value = quantity * unit price per transaction
                x1["line_total"] = x1[col_quantity1] * x1[col_unit1]
                # recency reference date is the most recent invoice date in the dataset
                reference_date = x1[col_date1].max() + pd.Timedelta(days=1)
                

                #rfm separate table dataframe
                rfm_df = x1.groupby(col_cust1).agg(
                    Recency=(col_date1, lambda x: (reference_date - x.max()).days),
                    Frequency=(col_invoice_no1, 'nunique'),
                    Monetary=("line_total", 'sum')
                ).reset_index()
                st.info("RFM features calculated successfully.")
                x1 = rfm_df[["Recency", "Frequency", "Monetary"]].dropna()
                # x1 = handle_outliers(x1).dropna()
                if x1.empty:
                    st.error("After handling outliers, there are no rows left. Please clean your CSV or pick other columns.")
                    st.stop()
                st.write(x1.head(100))
                x = x1.values
                
                
            run = st.button("🚀 Run clustering", type="primary")
            if not run:
                st.stop()
            algo, score, labels, no_cluster, factor2  = auto_best_clustering(x, None)
            if algo is None:
                st.error("Clustering failed. Please try manual mode or clean your data.")
                st.stop()
            else:
                st.success(f"Best algorithm: {algo} with silhouette score: {score:.2f}")
                st.write(f"Number of clusters: {no_cluster}")
                
                if rfm_direct:
                    df["cluster"] = np.nan
                    df.loc[x1.index, "cluster"] = labels
                    segment_map = build_segment_map(x1, labels, lower_is_better=[col_r1])
                    df["Customer_Importance"] = df["cluster"].map(segment_map)
                else:
                    rfm_df["cluster"] = np.nan
                    rfm_df.loc[x1.index, "cluster"] = labels
                    segment_map = build_segment_map(x1, labels, lower_is_better=["Recency"])
                    rfm_df["Customer_Importance"] = rfm_df["cluster"].map(segment_map)
                    df=rfm_df
                st.write(df)

                plot_cluster(x, mode, algo, score, factor2, labels, no_cluster, ["Recency", "Frequency", "Monetary"],segment_map=segment_map)
            



        else:
            st.warning("No automatic clustering possible. Switch to manual mode.")
            
elif manual_mode:
    st.info("Manual mode enabled. Please select features manually.")
    mode = "Manual_Mode" 
    original_features = [clean_original.get(f, f) for f in num_cols]
    clean_features = {original_clean.get(f,f) for f in num_cols}
    features = st.multiselect("Select features for clustering", options=original_features, default=original_features[:2],
                              max_selections=5)
   
    if len( features)<2:
        st.error("Please select at least 2 features for clustering.")
        st.stop()
    selected_cols = [original_clean.get(f, f) for f in features]
    x1 = df[selected_cols].dropna()
    if x1.empty:
        st.error("After dropping missing values, there are no rows left. Please clean your CSV or pick other columns.")
        st.stop()
    x1 = x1[x1 > 0]
    x1 = handle_outliers(x1).dropna()
    # x1 = x1.dropna()
    st.write(x1)
    algo = st.selectbox("Select clustering algorithm", options=["kmeans", "dbscan", "hierarchical", "minibatch_kmeans"], index=0)
    run = st.button("🚀 Run clustering", type="primary")
    if not run:
        st.stop()
    algo, score, labels, no_cluster, factor2 = auto_best_clustering(x1.values,algo)
    if algo is None:
        st.error("Clustering failed. Please try different features, algorithm or clean your data.")
        st.stop()
    segment_map = build_segment_map(x1, labels, lower_is_better=["recency"] if "Recency" in features else None)
    st.success(f"Silhouette Score for selected algorithm: {score:.2f} / No. of clusters: {no_cluster}")
    plot_cluster(x1.values, mode, algo, score, factor2, labels, no_cluster, features if len(features)<4 else ["PCA feature 1", "PCA feature 2", "PCA feature 3"], segment_map=segment_map)
    df["cluster"] = np.nan
    df.loc[x1.index, "cluster"] = labels
    df["Customer_Importance"] = df["cluster"].map(segment_map)
    st.write(df)
if mode in ["Basic_Mode", "RFM_Mode", "Manual_Mode"]:
    df = df.dropna(subset=["cluster"])
    # keep original column names for output
    final_df = df.copy()
    final_df.columns = [clean_original.get(c, c) for c in final_df.columns]

    csv_bytes = final_df.to_csv(index=False).encode("utf-8")

    st.download_button(
        "⬇️ Download Final CSV",
        data=csv_bytes,
        file_name="customer_segmentation_final.csv",
        mime="text/csv",
    )

      
            


# # Getting missing values reports using ydata_profiling
# from ydata_profiling import ProfileReport 
# if st.button(" 🧪 Generate Data Profiling Report", type="primary"):
#     profile = ProfileReport(df,title="Automatic Data Analysis ",
#                             explorative=True
#                             )
#     html(profile.to_html(), height=800, scrolling=True)



# run=st.button(" 🚀 Run Clustering", type="primary")

# Getting missing values reports using ydata_profiling
# from ydata_profiling import ProfileReport 

# profile = ProfileReport(df)
# html(profile.to_file("report.html"), height=800, scrolling=True)


    
   


