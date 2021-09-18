
def bestSum(target, numbers, dict):
    if target in dict:
        return dict[target]
    if target == 0:
        return []
    if target < 0:
        return None

    shortestCombination = None
    for num in numbers:
         resultCombination = bestSum(target - num, numbers, dict)
         if resultCombination is not None:
             resultCombination.append(num)
             if (shortestCombination is None or len(resultCombination) < len(shortestCombination)):
                shortestCombination = resultCombination

    dict[target] = shortestCombination
    return shortestCombination


def main():
    print(bestSum(8,[4,2,7],{}))



if __name__ == "__main__":
    main()
