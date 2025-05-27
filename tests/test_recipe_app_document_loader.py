from app.rag.recipe_app_document_loader import RecipeAppDocumentLoader

def test_load_documents():
    # 实例化 Loader，默认路径是 resources/documents/
    loader = RecipeAppDocumentLoader()

    # 加载文档
    loader.load_md()

