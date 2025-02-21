# -*- coding: utf-8 -*-
"""
Created on Mon Feb 10 08:35:16 2025

@author: 512051
"""
from pathlib import Path
from pprint import pprint
import math
from enum import Enum
from dataclasses import dataclass
from typing import BinaryIO, Any, Type
from io import BytesIO
import pandas as pd
import numpy as np
import struct
from abc import ABC, abstractmethod


df = pd.read_csv(p)


@dataclass
class Pattern:
    start: int
    delta: int
    length: int
    
    def matches(self, values: list[int]) -> bool:
        if len(values) != self.length:
            return False
        expected = [self.start + i * self.delta for i in range(self.length)]
        return all(a == b for a, b in zip(values, expected))
    
    def generate(self) -> list[int]:
        return [self.start + i * self.delta for i in range(self.length)]

@dataclass
class EmbeddedPattern:
    pattern: Pattern
    position: int  # where it occurs in the sequence
    repeat_count: int
    
    @property
    def total_length(self) -> int:
        return self.pattern.length * self.repeat_count


def detect_pattern(values: pd.Series) -> Pattern | None:
    """Detect if values form a pattern"""
    if len(values) < 2:
        return None
        
    # Check for constant delta
    diffs = values.diff()[1:].unique()
    if len(diffs) == 1:
        return Pattern(values.iloc[0], diffs[0], len(values))
        
    return None

def count_repetitions(series: pd.Series, 
                      pattern: Pattern, 
                      start_pos: int,
                      max_gap: int) -> int:
    """Count how many times pattern repeats, allowing for gaps"""
    count = 1
    pos = start_pos + pattern.length
    #pattern_values = pattern.generate()
    
    while pos < len(series):
        # Look for next occurrence within max_gap
        found = False
        for gap in range(max_gap):
            if pos + gap + pattern.length > len(series):
                break
                
            if pattern.matches(series.iloc[pos + gap:pos + gap + pattern.length].tolist()):
                count += 1
                pos = pos + gap + pattern.length
                found = True
                break
                
        if not found:
            break
            
    return count

def _optimize_patterns(self, patterns: list[EmbeddedPattern]) -> list[EmbeddedPattern]:
    """Remove overlapping patterns, keeping the most efficient ones"""
    if not patterns:
        return []
        
    # Sort by compression efficiency (could be more sophisticated)
    patterns.sort(key=lambda p: p.total_length * p.repeat_count, reverse=True)
    
    # Remove overlapping patterns
    final_patterns = []
    used_positions = set()
    
    for pattern in patterns:
        positions = set(range(pattern.position, 
                            pattern.position + pattern.total_length))
        if not positions & used_positions:  # No overlap
            final_patterns.append(pattern)
            used_positions.update(positions)
            
    return final_patterns


patterns = []
#series = series.iloc[:12].reset_index()
n = len(series)
min_length: int = 2
max_length: int = 150
max_gap: int = 2*max_length


# For each possible starting position
for start_pos in range(n - min_length + 1):
    # For each possible pattern length
    for length in range(min_length, max_length+1):
        #n - start_pos + 1
        if start_pos+length > n:
            break
        end_pos = min(n, start_pos + length)
        
        subset = series.iloc[start_pos:end_pos]
        #print(f"Start {start_pos}, Length {length}, End {end_pos}, n {n}")
        #print(subset)
        
#if False:
        # Try to detect if this is a pattern
        pattern = detect_pattern(subset)
        if pattern:
            # Look ahead for repetitions
            repeat_count = count_repetitions(series, pattern, 
                                                 start_pos, max_gap)
            if repeat_count > 1:  # Only store if it repeats
                patterns.append(EmbeddedPattern(pattern, start_pos, repeat_count))
                
                
                

