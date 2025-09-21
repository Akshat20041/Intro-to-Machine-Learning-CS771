import numpy as np
from sklearn.metrics import accuracy_score
from sklearn.svm import LinearSVC

# You are allowed to import any submodules of sklearn that learn linear models e.g. sklearn.svm etc
# You are not allowed to use other libraries such as keras, tensorflow etc
# You are not allowed to use any scipy routine other than khatri_rao

# SUBMIT YOUR CODE AS A SINGLE PYTHON (.PY) FILE INSIDE A ZIP ARCHIVE
# THE NAME OF THE PYTHON FILE MUST BE submit.py

# DO NOT CHANGE THE NAME OF THE METHODS my_fit, my_map, my_decode etc BELOW
# THESE WILL BE INVOKED BY THE EVALUATION SCRIPT. CHANGING THESE NAMES WILL CAUSE EVALUATION FAILURE

# You may define any new functions, variables, classes here
# For example, functions to calculate next coordinate or step length

################################
# Non Editable Region Starting #
################################
def my_fit(X_train, y_train):
    ################################
    #  Non Editable Region Ending  #
    ################################
    C_values = [1, 10]
    best_C = None
    best_val_acc = 0
    X_tr = my_map(X_train)  # Map the features first

    for C in C_values:
        clf = LinearSVC(loss="squared_hinge", C=C, max_iter=40000, tol=1e-1)
        clf.fit(X_tr, y_train)
        y_tr_pred = clf.predict(X_tr)
        train_acc = accuracy_score(y_train, y_tr_pred)
        if train_acc > best_val_acc:
            best_val_acc = train_acc
            best_C = C

    # Final training with best C
    clf = LinearSVC(loss="hinge", C=best_C, max_iter=10000, tol=1e-4)
    clf.fit(X_tr, y_train)

    # Extract weights and bias
    w_vector = clf.coef_.flatten()  # w (shape: n_features,)
    b_scalar = clf.intercept_[0]    # b (scalar)

    return w_vector, b_scalar  # ✅ Correct names, no NameError
################################
#  Non Editable Region Ending  #
################################





################################
# Non Editable Region Starting #
################################
def my_map( X ):
################################
#  Non Editable Region Ending  #
################################
    X = np.asarray(X)
    if X.shape[1] != 8:
        raise ValueError("Input must have 8 binary features.")

    d = 1 - 2 * X  # Convert {0,1} to {+1,-1}

    # Step 1: d0 to d7 (8 features)
    d_features = [d[:, i] for i in range(8)]

    # Step 2: pairwise products d0*d1 to d6*d7 (7 features)
    pairwise_products = [d[:, i] * d[:, i + 1] for i in range(7)]

    # Step 3: v = [d0,...,d7, d0d1,...,d6d7] → 15 features
    v = np.column_stack(d_features + pairwise_products)

    # Step 4: all unique cross terms vi * vj for i < j → 105 features
    cross_terms = []
    cross_indices = []
    for i in range(15):
        for j in range(i + 1, 15):
            cross_terms.append(v[:, i] * v[:, j])
            cross_indices.append((i, j))

    cross_terms = np.column_stack(cross_terms)

    # Step 5: remove 27 redundant features based on index pairs
    redundant_pairs = []

    for i in range(7):
      redundant_pairs.append((i, 8 + i))
    for i in range(1, 8):
      redundant_pairs.append((i, 7 + i))
    for i in range(7):
      redundant_pairs.append((i, i + 1))

# Always store pairs as (min, max)
    redundant_pairs = [tuple(sorted(pair)) for pair in redundant_pairs]

    redundant_indices = [idx for idx, pair in enumerate(cross_indices) if pair in redundant_pairs]


    # Remove redundant cross terms
    mask = np.ones(cross_terms.shape[1], dtype=bool)
    mask[redundant_indices] = False
    filtered_cross_terms = cross_terms[:, mask]

    # Step 6: Concatenate everything to get final features
    final_features = np.column_stack(d_features + pairwise_products + list(filtered_cross_terms.T))

    return final_features

################################
# Non Editable Region Starting #
################################
def my_decode( w ):
################################
#  Non Editable Region Ending  #
################################
    """
    Given a 65-dimensional linear model [w0, ..., w63, b], return
    four 64-dimensional vectors p, q, r, s (non-negative) representing
    delays for an Arbiter PUF that regenerate the same linear model.
    """
    k = 64
    wei = w[:k]
    b = w[k]

    # Step 1: Compute alpha and beta
    alpha = np.zeros(k)
    beta = np.zeros(k)

    alpha[0] = wei[0]
    beta[k - 1] = b

    # Back-calculate beta and alpha
    for i in reversed(range(1, k)):
        beta[i - 1] = wei[i] - alpha[i]
        alpha[i] = wei[i] - beta[i - 1]

    # Step 2: Compute delays with q_i = s_i = 0
    p = np.clip(alpha + beta, 0, None)  # p_i = αᵢ + βᵢ
    q = np.zeros(k)
    r = np.clip(alpha - beta, 0, None)  # rᵢ = αᵢ - βᵢ
    s = np.zeros(k)

    return p, q, r, s
