"""冻结商品服务的最小HTTP边界，不在导入时加载模型或启动服务器。"""

from typing import Literal

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, ConfigDict, Field, StrictInt

from memrecagent.product_service import BundleVersionError, ProductService


class RecommendationRequest(BaseModel):
    """只接受显式用户、版本与策略，不接受标签或任意历史时刻。"""

    model_config = ConfigDict(extra="forbid", str_strip_whitespace=True)
    user_id: str = Field(min_length=1, max_length=128)
    bundle_id: str = Field(min_length=1, max_length=128)
    strategy: Literal["popularity", "mf", "bpr", "bpr_cold_popularity"]
    k: StrictInt = Field(default=10, ge=1, le=50)


def create_app(service: ProductService) -> FastAPI:
    """启动层先验证并加载冻结包，再把只读推理对象交给路由。"""
    app = FastAPI(title="MemRecAgent Phase2", version="0.1.0")

    @app.get("/health")
    def health() -> dict:
        return service.health()

    @app.post("/recommendations")
    def recommendations(request: RecommendationRequest) -> dict:
        try:
            return service.recommend(
                request.user_id,
                bundle_id=request.bundle_id,
                strategy=request.strategy,
                k=request.k,
            )
        except BundleVersionError as exc:
            raise HTTPException(status_code=409, detail=str(exc)) from None
        except ValueError as exc:
            raise HTTPException(status_code=422, detail=str(exc)) from None

    return app
