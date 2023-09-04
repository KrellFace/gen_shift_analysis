
from enum import Enum, auto

class enum_MarioMetrics(Enum):
    
    #Main Fitness
    Playability = auto(),

    #Structural Metrics
    Contiguity = auto(),
    EnemyCount = auto(),
    RewardCount = auto(),
    EmptySpaceCount = auto(),
    BlockCount = auto(),
    PipeCount = auto(),
    Linearity = auto(),
    Density = auto(),
    ClearColumns = auto(),

    #Agent Extracted Metrics
    JumpCount = auto(),
    JumpEntropy = auto(),
    Speed = auto(),
    TimeTaken = auto(),
    TotalEnemyDeaths = auto(),
    KillsByStomp = auto(),
    MaxJumpAirTime = auto(),
    OnGroundRatio = auto(),
    AverageY = auto()

#Define the generators that will be evaluated

class enum_MarioGenerators(Enum):
    ge = auto(),
    hopper = auto(),
    notch = auto(),
    notchParam = auto(),
    ore = auto(),
    original = auto(),
    patternCount = auto(),
    patternOccur = auto(),
    patternWeightCount = auto()

