# -*- coding:UTF-8 -*-

"""
函数说明:
Parameters:
    InputString - 一段英文非空字符串
Returns:
    PassWord_string - 解密密码
"""
def getPassWordFromString(InputString):
    PassWord = []
    PassWord_string = ""
    WordDict = {}
    InputString = InputString.lower()
    
    for i in InputString:
        if i in WordDict:
            WordDict[i]+=1
        else:
            WordDict[i] = 1
    
    while sum(list(WordDict.values())) > 0:
        if "x" in WordDict and WordDict["x"] > 0:
            if 's' or 'i' or 'x' not in WordDict.keys():
                PassWord_string = "InputString is Err, can't get PassWord"
                return PassWord_string
            for i in range(WordDict["x"]):
                WordDict["s"] -= 1
                WordDict["i"] -= 1
                WordDict["x"] -= 1
                PassWord.append(6)
        if "z" in WordDict and WordDict["z"] > 0:
            if 'z' or 'e' or 'r' or 'o' not in WordDict.keys():
                PassWord_string = "InputString is Err, can't get PassWord"
                return PassWord_string
            for i in range(WordDict["z"]):
                WordDict["z"] -= 1
                WordDict["e"] -= 1
                WordDict["r"] -= 1
                WordDict["o"] -= 1
                PassWord.append(0)
        if "w" in WordDict and WordDict["w"] > 0:
            if 't' or 'w' or 'o' not in WordDict.keys():
                PassWord_string = "InputString is Err, can't get PassWord"
                return PassWord_string
            for i in range(WordDict["w"]):
                WordDict["t"] -= 1
                WordDict["w"] -= 1
                WordDict["o"] -= 1
                PassWord.append(2)
        if "u" in WordDict and WordDict["u"] > 0:
            if 'f' or 'o' or 'u' or 'r' not in WordDict.keys():
                PassWord_string = "InputString is Err, can't get PassWord"
                return PassWord_string
            for i in range(WordDict["u"]):
                WordDict["f"] -= 1
                WordDict["o"] -= 1
                WordDict["u"] -= 1
                WordDict["r"] -= 1
                PassWord.append(4)
        if "r" in WordDict and WordDict["r"] > 0:
            if 't' or 'h' or 'r' or 'e' not in WordDict.keys():
                PassWord_string = "InputString is Err, can't get PassWord"
                return PassWord_string
            for i in range(WordDict["r"]):
                WordDict["t"] -= 1
                WordDict["h"] -= 1
                WordDict["r"] -= 1
                WordDict["e"] -= 2
                PassWord.append(3)
        if "o" in WordDict and WordDict["o"] > 0:
            if 'o' or 'n' or 'e' not in WordDict.keys():
                PassWord_string = "InputString is Err, can't get PassWord"
                return PassWord_string
            for i in range(WordDict["o"]):
                WordDict["o"] -= 1
                WordDict["n"] -= 1
                WordDict["e"] -= 1
                PassWord.append(1)
        if "f" in WordDict and WordDict["f"] > 0:
            if 'f' or 'i' or 'v' or 'e' not in WordDict.keys():
                PassWord_string = "InputString is Err, can't get PassWord"
                return PassWord_string
            for i in range(WordDict["f"]):
                WordDict["f"] -= 1
                WordDict["i"] -= 1
                WordDict["v"] -= 1
                WordDict["e"] -= 1
                PassWord.append(5)
        if "s" in WordDict and WordDict["s"] > 0:
            if 's' or 'v' or 'n' or 'e' not in WordDict.keys():
                PassWord_string = "InputString is Err, can't get PassWord"
                return PassWord_string
            for i in range(WordDict["s"]):
                WordDict["s"] -= 1
                WordDict["v"] -= 1
                WordDict["n"] -= 1
                WordDict["e"] -= 2
                PassWord.append(7)
        if "t" in WordDict and WordDict["t"] > 0:
            if 'e' or 'i' or 'g' or 'h' or 't' not in WordDict.keys():
                PassWord_string = "InputString is Err, can't get PassWord"
                return PassWord_string
            for i in range(WordDict["t"]):
                print(WordDict)
                WordDict["e"] -= 1
                WordDict["i"] -= 1
                WordDict["g"] -= 1
                WordDict["h"] -= 1
                WordDict["t"] -= 1
                PassWord.append(8)
        if "i" in WordDict and WordDict["i"] > 0:
            if 'n' or 'i' or 'e' not in WordDict.keys():
                PassWord_string = "InputString is Err, can't get PassWord"
                return PassWord_string
            for i in range(WordDict["i"]):
                WordDict["n"] -= 2
                WordDict["i"] -= 1
                WordDict["e"] -= 1
                PassWord.append(9)
    
    PassWord.sort()
    for i in PassWord:
        PassWord_string+=str(i)
    
    return PassWord_string

"""
简易解密
"""
if __name__ == '__main__':
    InputString = input()
    PassWord_string = getPassWordFromString(InputString)
    print(PassWord_string)
    
