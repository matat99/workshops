"""
Vector analogies: A is to B as C is to ?
This demonstrates that directions in vector space represent concepts.

The formula: B - A + C = D
Example: woman - man + king ≈ queen
"""

from load_model import load_word2vec_model

# Load the pre-trained Word2Vec model from cache
model = load_word2vec_model()

def A_is_to_B_as_C_is_to(a, b, c, topn=1):
    """
    Solve analogies: A is to B as C is to ?
    
    Args:
        a: The first term (what we subtract)
        b: The second term (what we add)
        c: The third term (what we add)
        topn: Number of results to return
    
    Returns:
        The most likely completion(s) of the analogy
    """
    # Convert single items to lists for consistency
    a = a if isinstance(a, list) else [a]
    b = b if isinstance(b, list) else [b]
    c = c if isinstance(c, list) else [c]
    
    # positive terms get added (B and C)
    # negative terms get subtracted (A)
    res = model.most_similar(positive=b + c, negative=a, topn=topn)
    
    if len(res):
        if topn == 1:
            return res[0][0]
        return [x[0] for x in res]
    return None

# Classic examples
print("=== Gender Analogies ===")
print(f"man is to woman as king is to {A_is_to_B_as_C_is_to('man', 'woman', 'king')}")
print(f"man is to woman as actor is to {A_is_to_B_as_C_is_to('man', 'woman', 'actor')}")
print()

# Geography - capitals
print("=== Geography: Capitals ===")
for country in ['Italy', 'France', 'India', 'China']:
    capital = A_is_to_B_as_C_is_to('Germany', 'Berlin', country)
    print(f"Germany is to Berlin as {country} is to {capital}")
print()

# Company products
print("=== Company Products ===")
for company in ['Google', 'IBM', 'Boeing', 'Microsoft', 'Samsung']:
    products = A_is_to_B_as_C_is_to(
        ['Starbucks', 'Apple'], 
        ['Starbucks_coffee', 'iPhone'], 
        company, 
        topn=3
    )
    print(f"{company} -> {', '.join(products)}")
print()
