import os


def readFile(filename):
    current_directory_path = os.path.dirname(os.path.abspath(__file__))
    data_path = os.path.join(current_directory_path, "..\..\data\\")
    file_path = os.path.join(data_path, filename)

    file_details = {}
    # Getting the file details
    with open(file_path, "r") as f:
        while True:
            line = f.readline().strip()

            if line.startswith("NAME"):
                file_details["Name"] = line.split(":")[1]

            elif line.startswith("DIMENSION"):
                file_details["Dimensions"] = int(line.split(":")[1])

            elif line.startswith("EDGE_WEIGHT_TYPE"):
                file_details["Edge Weight Type"] = line.split(":")[1].strip()

            elif line.startswith("NODE_COORD_SECTION"):
                break

            else:
                continue

        if file_details["Edge Weight Type"] != "EUC_2D":
            print(
                "Edge weight types are not Euclidean distances in 2D. Please try with another file."
            )

        else:
            # Creating a dictionary of the nodes along with their coordinates
            cities = {}
            for i in range(file_details["Dimensions"]):
                node, x, y = map(float, f.readline().strip().split())
                cities[int(node)] = (x, y)

        return cities