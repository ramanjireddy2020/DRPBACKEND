"""
DRP_Main — Main entry points.

This module provides:
1. A CLI entry point for Databricks job execution (main)
2. A FastAPI application entry point for serving the API (app)
"""
import argparse
import sys
import os

# Ensure the package root is in the path for Databricks runtime
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def main():
    """
    CLI entry point for Databricks job execution.
    Usage: main --catalog <catalog> --schema <schema>
    """
    from databricks.sdk.runtime import spark

    parser = argparse.ArgumentParser(
        description="DRP_Main Databricks job with catalog and schema parameters",
    )
    parser.add_argument("--catalog", required=True, help="Target catalog name")
    parser.add_argument("--schema", required=True, help="Target schema name")
    args = parser.parse_args()

    # Set the default catalog and schema
    spark.sql(f"USE CATALOG {args.catalog}")
    spark.sql(f"USE SCHEMA {args.schema}")

    print(f"DRP_Main job running in catalog={args.catalog}, schema={args.schema}")
    print("Databricks environment is ready for InnoDD API operations.")


def run_fastapi():
    """
    Run the FastAPI application server.
    Used for local development and Databricks model serving.
    """
    import uvicorn
    from DRP_Main.app.main import app

    uvicorn.run(app, host="0.0.0.0", port=8080)


if __name__ == "__main__":
    # If called directly, check if we should run the API or the CLI job
    if "--api" in sys.argv:
        sys.argv.remove("--api")
        run_fastapi()
    else:
        main()