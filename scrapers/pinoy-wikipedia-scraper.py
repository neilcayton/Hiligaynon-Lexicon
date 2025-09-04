#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import sys
import json
import time
import random
import logging
import re
from typing import List, Dict, Any, Optional, Set
from bs4 import BeautifulSoup
import requests
import string
from urllib.parse import urljoin

# In the original script, this was an import from a utils file.
# For simplicity, we define it here.
OUTPUT_DIR = os.path.join(os.path.dirname(__file__), '../data/lexicon')
os.makedirs(OUTPUT_DIR, exist_ok=True)


class BaseScraper:
    """A simple base class for scrapers."""
    def __init__(self, source_name: str, base_url: str, output_dir: str, delay: float):
        self.source_name = source_name
        self.base_url = base_url
        self.output_dir = output_dir
        self.delay = delay
        self.logger = logging.getLogger(self.__class__.__name__)


class PinoyDictionaryScraper(BaseScraper):
    """
    Scraper for the PinoyDictionary dictionaries.
    Supports multiple dictionaries: Tagalog, Cebuano, Hiligaynon, and Ilocano
    URLs: https://[dictionary].pinoydictionary.com
    """

    def __init__(self, output_dir: str = OUTPUT_DIR, delay: float = 2.0, dictionary: str = 'hiligaynon'):
        """Initialize the Pinoy Dictionary scraper
        
        Args:
            output_dir: Directory to save output files
            delay: Delay between requests in seconds
            dictionary: Dictionary to scrape ('tagalog', 'cebuano', 'hiligaynon', or 'ilocano')
        """
        self.available_dictionaries = ["tagalog", "cebuano", "hiligaynon", "ilocano"]
        if dictionary not in self.available_dictionaries:
            raise ValueError(f"Dictionary must be one of {self.available_dictionaries}")
            
        self.dictionary = dictionary
        base_url = f"https://{dictionary}.pinoydictionary.com"
        
        super().__init__(
            source_name=f"PinoyDictionary-{dictionary.capitalize()}",
            base_url=base_url,
            output_dir=output_dir,
            delay=delay
        )

    def get_words(self, starting_letter: str) -> List[Dict[str, Any]]:
        """
        Get words for a specific starting letter with pagination, extracting detailed information.
        
        Args:
            starting_letter: Letter to scrape words for
            
        Returns:
            List of word entries with structured data.
        """
        page_number = 1
        words = []
        
        while True:
            url = f"https://{self.dictionary}.pinoydictionary.com/list/{starting_letter}/{page_number}/"
            self.logger.info(f"Reading {url}...")
            time.sleep(self.delay)
            
            try:
                page = requests.get(url)
                page.raise_for_status()  # Raise an HTTPError for bad responses (4xx or 5xx)
                soup = BeautifulSoup(page.content, "html.parser")
                
                if soup.find(class_="page-not-found"):
                    break
                
                word_groups = soup.find_all(class_="word-group")
                if not word_groups:
                    break
                    
                for word_group in word_groups:
                    word_html = word_group.find(class_="word")
                    if not word_html:
                        continue
                        
                    word_entry_html = word_html.find(class_="word-entry")
                    if not word_entry_html or not word_entry_html.find("a"):
                        continue
                        
                    word = word_entry_html.find("a").text.strip()
                    link = urljoin(self.base_url, word_entry_html.find("a").get("href"))
                    
                    # Extract part of speech, which is usually in a <p> tag next to the word
                    part_of_speech_elem = word_html.find("p")
                    part_of_speech = part_of_speech_elem.text.strip() if part_of_speech_elem else ""
                    
                    definition_html = word_group.find(class_="definition")
                    if not definition_html:
                        continue

                    # --- New parsing logic for structured content ---
                    english_translation = ""
                    examples = []

                    # The text of the definition div contains both the main meaning and examples
                    # We use a separator to avoid words running together.
                    full_definition_text = definition_html.get_text(separator=' ', strip=False)

                    # Examples are marked with asterisks (*). We split the text by these examples.
                    # The first part of the split is the main definition.
                    split_by_examples = re.split(r'\s*\*', full_definition_text)
                    
                    if split_by_examples:
                        english_translation = split_by_examples[0].strip()

                    # Find all Hiligaynon and English example pairs
                    # A Hiligaynon example is text enclosed in asterisks.
                    # Its English translation is the text that follows it.
                    matches = list(re.finditer(r'\*([^*]+)\*', full_definition_text))
                    for i, match in enumerate(matches):
                        hiligaynon_example = match.group(1).strip()
                        
                        # The English translation is the text between the end of this match 
                        # and the start of the next one (or the end of the string).
                        start_index = match.end(0)
                        end_index = matches[i + 1].start(0) if i + 1 < len(matches) else len(full_definition_text)
                        
                        english_example_raw = full_definition_text[start_index:end_index].strip()
                        
                        # Clean up the translation, removing leading punctuation and trailing info.
                        english_example = re.sub(r'^\s*[\.,;]?', '', english_example_raw).strip()
                        # Often there is a "(see ...)" part at the end, we can remove it.
                        english_example = re.split(r'\s*\(see', english_example)[0].strip()

                        if hiligaynon_example and english_example:
                            examples.append({
                                'hiligaynon': hiligaynon_example,
                                'english': english_example
                            })
                    
                    # Create the final structured entry for the word
                    entry = {
                        'word': word,
                        'part_of_speech': part_of_speech,
                        'english_translation': english_translation,
                        'examples': examples,
                        'link': link
                    }
                    words.append(entry)
                    
                self.logger.info(f"Found {len(words)} words on page {page_number} for letter '{starting_letter}'")
                page_number += 1
                
            except requests.exceptions.RequestException as e:
                self.logger.error(f"Error fetching page {url}: {str(e)}")
                break
            except Exception as e:
                self.logger.error(f"An unexpected error occurred processing page {url}: {str(e)}")
                if 'word_group' in locals():
                    self.logger.error(f"Content that caused error: {word_group.prettify()}")
                break
                
        return words

    def scrape(self, max_letters: Optional[Any] = None, max_letters_per_dict: Optional[int] = None, dictionaries: Optional[List[str]] = None) -> List[Dict[str, Any]]:
        """
        Scrape the Pinoy Dictionary website
        
        Args:
            max_letters: List of specific letters to scrape, e.g. ['a', 'b', 'c']
            max_letters_per_dict: Maximum number of first letters to scrape per dictionary
            dictionaries: List of dictionaries to scrape. If None, uses the one specified in __init__
            
        Returns:
            List of dictionaries containing structured word data
        """
        starting_letters = list(string.ascii_lowercase)
        if isinstance(max_letters, list):
            starting_letters = [l for l in starting_letters if l in max_letters]
        elif isinstance(max_letters, int):
            starting_letters = starting_letters[:max_letters]
        elif max_letters_per_dict:
            starting_letters = starting_letters[:max_letters_per_dict]
            
        all_words = []
        
        dicts_to_scrape = [self.dictionary]
        if dictionaries:
            valid_dicts = [d for d in dictionaries if d in self.available_dictionaries]
            if valid_dicts:
                dicts_to_scrape = valid_dicts
            else:
                self.logger.warning(f"No valid dictionaries specified. Using {self.dictionary}")
        
        for dictionary in dicts_to_scrape:
            current_dict = self.dictionary
            self.dictionary = dictionary
            self.base_url = f"https://{dictionary}.pinoydictionary.com"
            self.source_name = f"PinoyDictionary-{dictionary.capitalize()}"
            self.logger.info(f"Scraping {dictionary} dictionary")
            
            try:
                for starting_letter in starting_letters:
                    self.logger.info(f"Processing letter: {starting_letter} for {dictionary}")
                    letter_words = self.get_words(starting_letter)
                    all_words.extend(letter_words)
                    self.logger.info(f"Found {len(letter_words)} words for letter '{starting_letter}'")
                    time.sleep(self.delay / 2) # Smaller delay between letters
            except Exception as e:
                self.logger.error(f"Error occurred during scraping {dictionary}: {str(e)}")
            
            self.dictionary = current_dict
                
        self.logger.info(f"Total words collected: {len(all_words)}")
        return all_words


def comma_separated(string: str) -> List[str]:
    """Split a comma-separated string into a list"""
    return [item.strip() for item in string.split(",")] if string else []


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    import argparse
    parser = argparse.ArgumentParser(description='Pinoy Dictionary Scraper')
    parser.add_argument('--dictionary', type=str, default='hiligaynon', help='Default dictionary to scrape (tagalog, cebuano, hiligaynon, ilocano)')
    parser.add_argument('--dictionaries', type=comma_separated, help='Comma-separated list of dictionaries to scrape (e.g., tagalog,cebuano)')
    parser.add_argument('--max_letters', type=int, help='Maximum number of letters to scrape (e.g., 3 for a,b,c)')
    parser.add_argument('--letters', type=comma_separated, help='Comma-separated list of specific letters to scrape (e.g., a,b,c)')
    parser.add_argument('--letter', type=str, help='Single specific letter to scrape (for testing)')
    parser.add_argument('--output_file', type=str, help='Output JSON file path')
    args = parser.parse_args()
    
    scraper = PinoyDictionaryScraper(dictionary=args.dictionary)
    
    words = []
    if args.letter:
        print(f"Scraping only words starting with '{args.letter}' from {args.dictionary} dictionary...")
        words = scraper.get_words(args.letter)
    else:
        print("Starting comprehensive scrape...")
        words = scraper.scrape(
            max_letters=args.letters or args.max_letters,
            dictionaries=args.dictionaries
        )
    
    if words:
        # Use the specified output file or a default name
        output_file = args.output_file or "pinoy_dictionary_output.json"
        
        # Ensure the output directory exists
        output_dir = os.path.dirname(output_file)
        if output_dir:
            os.makedirs(output_dir, exist_ok=True)

        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(words, f, indent=4, ensure_ascii=False)
        print(f"Scraped {len(words)} words and saved to {output_file}")
    else:
        print("No words were scraped.")

