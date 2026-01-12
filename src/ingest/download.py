"""Download Food.com dataset via Kaggle API."""

from pathlib import Path
from kaggle.api.kaggle_api_extended import KaggleApi
from src.app.logging_config import get_logger

logger = get_logger(__name__)


def download_foodcom_dataset(output_dir: Path) -> None:
    """Download Food.com dataset via Kaggle API.

    Dataset: shuyangli94/food-com-recipes-and-user-interactions
    Files:
      - RAW_recipes.csv (~230MB, 231K recipes)
      - RAW_interactions.csv (~420MB, 1.1M interactions)

    Args:
        output_dir: Directory to download and extract files to

    Raises:
        Exception: If Kaggle authentication fails or download fails
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    logger.info("Initializing Kaggle API client")
    api = KaggleApi()
    api.authenticate()

    dataset_name = "shuyangli94/food-com-recipes-and-user-interactions"
    logger.info(
        "Downloading dataset",
        dataset=dataset_name,
        output_dir=str(output_dir)
    )

    # Download and unzip
    api.dataset_download_files(
        dataset_name,
        path=output_dir,
        unzip=True
    )

    # Verify files exist
    recipes_csv = output_dir / "RAW_recipes.csv"
    interactions_csv = output_dir / "RAW_interactions.csv"

    if not recipes_csv.exists():
        raise FileNotFoundError(f"Expected file not found: {recipes_csv}")
    if not interactions_csv.exists():
        raise FileNotFoundError(f"Expected file not found: {interactions_csv}")

    logger.info(
        "Dataset downloaded successfully",
        recipes_csv=str(recipes_csv),
        interactions_csv=str(interactions_csv),
        recipes_size_mb=round(recipes_csv.stat().st_size / 1024 / 1024, 2),
        interactions_size_mb=round(interactions_csv.stat().st_size / 1024 / 1024, 2)
    )
