# ======================================
# CELL 2 - DATA LOADING
# ======================================
# Load the WUSTL-IIoT-2021 dataset.
# Only drop pure identifier / timestamp columns.
# Feature selection will handle the rest scientifically.

df = pd.read_csv(r"D:\cyberr\wustl_iiot_2021\wustl_iiot_2021.csv")

print("Original Shape:", df.shape)
print("Columns:", list(df.columns))

# Drop only non-numeric identifiers / timestamps
id_cols = ['StartTime', 'LastTime', 'SrcAddr', 'DstAddr',
           'Sport', 'Dport', 'Proto', 'Dir', 'state']
df = df.drop(columns=[c for c in id_cols if c in df.columns], errors='ignore')
print(f"\nAfter dropping identifiers: {df.shape}")

# Handle any remaining NaN / Inf
df.replace([np.inf, -np.inf], np.nan, inplace=True)
df.fillna(0, inplace=True)

# Encode multi-class target (Traffic)
le = LabelEncoder()
df['Traffic'] = le.fit_transform(df['Traffic'])

# Separate features and target
X_full = df.drop(['Traffic', 'Target'], axis=1, errors='ignore')
y_full = df['Traffic']

print(f"Feature matrix: {X_full.shape}")
print(f"Number of classes: {y_full.nunique()}")
print(f"Class distribution:\n{y_full.value_counts().sort_index()}")
