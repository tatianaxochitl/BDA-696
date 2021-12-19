from hw4_data import get_test_data_set
from midterm import process_dataframe


def main():
    df, pred, resp = get_test_data_set("titanic")
    process_dataframe(df, pred, resp)


if __name__ == "__main__":
    main()
