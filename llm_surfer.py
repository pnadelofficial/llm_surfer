from duckduckgo_search import DDGS
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline, BitsAndBytesConfig
from sentence_transformers import SentenceTransformer
import torch
from jsonformer import Jsonformer
from functools import partial
import re
from langchain.text_splitter import TokenTextSplitter
from selenium import webdriver
from selenium.webdriver.firefox.service import Service
from selenium.webdriver.common.by import By
from selenium.webdriver.support.wait import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from bs4 import BeautifulSoup
import requests
import pypdf
import pandas as pd
import time
from datetime import datetime
from tqdm import tqdm
import requests 

bnb_config = BitsAndBytesConfig(
    load_in_4bit=True, 
    bnb_4bit_use_double_quant=True, 
    bnb_4bit_quant_type="nf4", 
    bnb_4bit_compute_dtype=torch.bfloat16
)

service = Service() 
options = webdriver.FirefoxOptions()
options.add_argument("--headless")
options.add_argument("--mute-audio")
options.add_argument('--no-sandbox')
options.add_argument('--disable-dev-shm-usage')
options.add_argument("--disable-blink-features=AutomationControlled")
options.add_argument("user-agent=Mozilla/5.0 (Windows NT 10.0; Win64; x64; rv:90.0) Gecko/20100101 Firefox/90.0")

class Searcher:
    def __init__(self, query, max_results=10, search_engine='congress'): 
        self.query = query
        self.max_results = max_results
        self.search_engine = search_engine
        
    def _ddgs(self):
        results = []
        with DDGS() as ddgs:
            for r in ddgs.text(self.query, max_results=self.max_results):
                results.append(r)
        self.results = results

    def _congress(self):
        self.webdriver = webdriver.Firefox(service=service, options=options)
        if 'OR' in self.query:
            parts = []
            for i, q in enumerate(self.query.split('OR')):
                part = q.strip().replace('"', '')
                if ' ' in part:
                    part = part.replace(' ', '+')
                part = "%5C%22" + part + "%5C%22+"
                if i != len(self.query.split('OR'))-1:
                    part = part + "OR+"
                parts.append(part)
            search = ''.join(parts)
            url = f"https://www.congress.gov/search?q=%7B%22congress%22%3A%22all%22%2C%22search%22%3A%22{search}%22%2C%22bill-status%22%3A%22law%22%7D"
        else:
            if len(self.query.split(' ')) > 1:
                url = f"https://www.congress.gov/search?q=%7B%22congress%22%3A%22all%22%2C%22source%22%3A%22all%22%2C%22search%22%3A%22{'+'.join(self.query.split(' '))}%22%2C%22bill-status%22%3A%22law%22%7D"
            else:
                url = f"https://www.congress.gov/search?q=%7B%22congress%22%3A%22all%22%2C%22source%22%3A%22all%22%2C%22search%22%3A%22{self.query}%22%2C%22bill-status%22%3A%22law%22%7D"
        time.sleep(5)
        self.webdriver.get(url)
        time.sleep(10)
        
        more_pages = True
        self.results = []
        current_page = 1
        max_pages = int(self.webdriver.find_elements(By.CLASS_NAME, "results-number")[-1].text.split('of')[-1].strip().replace(',',''))//100 + 1
        while more_pages:
            ol = self.webdriver.find_element(By.TAG_NAME, "ol")
            lis = ol.find_elements(By.XPATH, "//li[@class='expanded']")
            for li in lis:
                soup = BeautifulSoup(li.get_attribute('innerHTML'), "html.parser")
                raw = soup.a['href']
                title = soup.find('span', class_="result-title").get_text().strip()
                congress = raw.split('/')[2][:3]
                bill_no = raw.split('?')[0].split('/')[-1]
                self.results.append((title, congress, bill_no))
            current_page+=1
            if current_page > max_pages:
                more_pages = False
            else:
                self.webdriver.get(url+f"&page={current_page}")
        self.webdriver.close()

    def _congress_scrape(self, tup):
        congress, bill_no = tup
        url = f"https://api.congress.gov/v3/bill/{congress}/hr/{bill_no}/text"
        session = requests.Session()
        session.params = {"format": 'json'}
        session.headers.update({"x-api-key": '5VhOEr0OcuyhGgZlGRQX26b0av7Jp5JE8qDeStCb'})
        
        res = session.get(url)
        if res:
            data = res.json()
            dict_list = [d for d in data['textVersions'] if len(d['formats']) > 0] 
            res_ex = session.get([d for d in dict_list[-1]['formats'] if d['url'].endswith('xml') or d['url'].endswith('htm')][-1]['url'])
            soup = BeautifulSoup(res_ex.text, features='xml') 
            text = soup.get_text()
            poss_title = soup.find("dc:title")
            if poss_title:
                title = poss_title.get_text()
            elif ('xml' in res_ex.url) and not poss_title: 
                try:
                    title = soup.find('official-title').get_text().replace('\n\t\t', '').replace('  ', ' ')
                except:
                    # error from api
                    print(tup, 'law not found. Skipping...')
            else:
                try:
                    title = re.search(r'An Act(.*?\.)', text, re.DOTALL).group(1).strip().replace('\n', '').replace('     ', '')
                except:
                    title = re.search(r'H\. R\.\s+\d+(.*?\.)', text, re.DOTALL).group(1).strip().replace('\n', '').replace('     ', '').replace('   ', '')
            return dict_list[-1]['formats'][-1]['url'], text, title, dict_list[-1]['date'].split('-')[0]
        else:
            # error from api
            print(tup, "law not found. Skipping...")
            return ''

    def _openalex(self):
        base_url = self.open_alex_base_url.format(self.query) 
        num_results = requests.get(self.open_alex_base_url.format(self.query)).json()['meta']['count'] 
        
        base_url = self.open_alex_base_url.format(self.query) + '&page={}'
                
        page = 1
        has_more_pages = True
        fewer_than_10k_results = True
        full_results = []
        print("Retrieving results from OpenAlex")
        while has_more_pages and fewer_than_10k_results: #and (len(full_results) < self.max_results+10)
            print(f"Reading OpenAlex Page {page}", end='\r')
            url = base_url.format(page)
            page_with_results = requests.get(url).json()
            
            results = page_with_results['results']
            full_results.extend(results)
            if len(full_results) > self.max_results*20:
                break
    
            page += 1
            per_page = page_with_results['meta']['per_page']
            has_more_pages = len(results) == per_page
            fewer_than_10k_results = per_page * page <= 10000

        self.results = [{'title':r['title'], 'href':r['locations'][0]['landing_page_url']} for r in full_results]
    
    def search(self):
        if self.search_engine == "ddg":
            self._ddgs()
        elif self.search_engine == "congress":
            self._congress()
        elif self.search_engine == 'openalex':
            self.open_alex_base_url = "https://api.openalex.org/works?filter=default.search:{},open_access.is_oa:true&sort=relevance_score:desc"
            self._openalex()

    def scrape_from_url(self, url):
        service = Service()
        self.webdriver = webdriver.Firefox(service=service, options=options)
        time.sleep(5)
        self.webdriver.get(url)
        # for crs pdfs, could be others without this format....
        if 'pdf' in url:
            pages = self.webdriver.find_elements(By.CLASS_NAME, 'page')
            full_text = ''
            for page in pages:
                full_text += page.text
                self.webdriver.find_element(By.ID, 'next').click()
        else:
            body_html = self.webdriver.find_element(By.TAG_NAME, 'body').get_attribute('innerHTML')
            soup = BeautifulSoup(body_html, 'html.parser')
            full_text = ' '.join([d.get_text().replace('\n', '') for d in soup.find_all('div', {'class':'section'})])
            if full_text == '':
                full_text = ' '.join([p.get_text().replace('\n', '') for p in soup.find_all('p')]) 
        self.webdriver.close()
        return url, full_text

    def __call__(self):
        self.search()
        self.results_list = []
        res_to_loop = self.results[:self.max_results] if len(self.results) > self.max_results else self.results
        with tqdm(total=self.max_results) as pbar:
            for i, result in enumerate(res_to_loop):
                if len(self.results_list) >= self.max_results:
                    break

                if self.search_engine == 'congress':
                    try:
                        title = result[0]
                        url, text, alt_title, date = self._congress_scrape(result[1:])
                        if text == '':
                            continue
                        self.results_list.append((url, title, text, date, alt_title))
                    except:
                        pass
                else:
                    url, text = self.scrape_from_url(result['href'])
                    
                    title = result['title']

                    if text == '':
                        print('No text found. Skipping...')
                        for r in self.results:
                            if r not in res_to_loop:
                                res_to_loop.append(r)
                                print('New result added.')
                                break
                    else:
                        self.results_list.append((url, title, text, 'No date found', ''))
                pbar.update(1)
        return self.results_list

class Embedder:
    def __init__(self, result, model_path='/cluster/tufts/tuftsai/models/BAAI_m3', device='cuda', chunk_size=125, chunk_overlap=25, batch_size=64):
        self.result = result
        self.model_path = model_path
        self.device = device
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.batch_size = batch_size
        
        self.model = SentenceTransformer(self.model_path, device=self.device)
        self.text_splitter = TokenTextSplitter(chunk_size=self.chunk_size, chunk_overlap=self.chunk_overlap)

    def _read_results(self):
        self.doc_dict = {(self.result[0], self.result[1]):self.result[2]}

    def _split_into_chunks(self):
        self.chunk_dict = {tup:self.text_splitter.split_text(text) for tup, text in self.doc_dict.items()}

    def _encode(self):
        # SHOULD ADD LOGIC FOR TRACKING DOCUMENT METADATA
        self.all_chunks = [item for sublist in self.chunk_dict.values() for item in sublist]
        embeddings = self.model.encode(self.all_chunks, batch_size=self.batch_size, device=self.device, convert_to_tensor=True, normalize_embeddings=True)
        self.embeddings = embeddings

    def __call__(self):
        self._read_results()
        self._split_into_chunks()
        self._encode()
        return self.embeddings, self.all_chunks

class RAG:
    def __init__(self, query, embeddings, chunks, model=None, top_k=5, model_path="/cluster/tufts/tuftsai/models/BAAI_m3", device='cuda'):
        self.query = query
        self.embeddings = embeddings
        self.chunks = chunks
        self.top_k = top_k
        self.model_path = model_path
        self.device = device
        self.retrieval_instruction = "Represent this sentence for searching relevant passages:"
        self.model = model if not model else SentenceTransformer(self.model_path, device=self.device)

    def _sim_search(self):
        query_embedding = self.model.encode(self.retrieval_instruction+self.query, device=self.device, convert_to_tensor=True, normalize_embeddings=True)
        context_idxs = (self.embeddings.to(self.device) @ query_embedding.to(self.device)).argsort().cpu().numpy()
        self.context = [self.chunks[i] for i in context_idxs[::-1][:self.top_k]]
        return "\n".join(self.context), self.context

    def __call__(self):
        return self._sim_search()

class LLMSurfer:
    def __init__(self, llm_name, research_goal, base_prompt, json_schema, query, max_results, search_engine='congress', device='cuda'):
        self.llm_name = llm_name
        self.research_goal = research_goal
        self.base_prompt = base_prompt
        self.json_schema = json_schema
        self.query = query.strip()
        self.max_results = max_results
        self.search_engine = search_engine
        self.device = torch.device(device)

        self.tokenizer = AutoTokenizer.from_pretrained(self.llm_name,)
        self.model = AutoModelForCausalLM.from_pretrained(self.llm_name, device_map='auto', quantization_config=bnb_config)
        self.jsonformer_partial = partial(Jsonformer, self.model, self.tokenizer, self.json_schema, max_number_tokens=200, max_string_token_length=200)

    def get_results(self):
        self.results = self.searcher()
    
    def process_one(self, i, result):
        url = result[0]
        title = result[1] 
        print(f"Reading URL: {url}")
        embedder = Embedder(result)
        embeddings, chunks = embedder()
        if len(chunks) == 0:
            print("No content found on this site. Passing...")
            print('--'*50)
            return None
        res_str, context_chunks = RAG(self.query, embeddings, chunks, model=embedder.model)()
        del embedder.model
        filled_prompt = self.base_prompt.format(research_goal=self.research_goal, url=re.escape(url), title=re.escape(title), text=re.escape(res_str))
        filled_prompt_list = [{"role":"user", "content":filled_prompt}]
        input_to_jsonformer = self.tokenizer.apply_chat_template(
            filled_prompt_list,
            tokenize=False,
            add_generation_prompt=True
        )
        return self.jsonformer_partial(input_to_jsonformer)(), context_chunks

    def __call__(self, to_excel=True, num_rel_chunks=5):
        self.searcher = Searcher(query=self.query, max_results=self.max_results, search_engine=self.search_engine)
        print(f"Collecting links from {self.searcher.search_engine}\r", flush=True)
        self.get_results()
        print(f"{len(self.results)} links collected out of {self.max_results}. The rest are unreachable")
        rel_docs = {}
        print('--'*50)
        for i, result in enumerate(self.results):
            if result[0] not in rel_docs:
                print(f"Webpage #{i+1}")
                date = result[-2]
                alt_title = result[-1]
                out, chunks = self.process_one(i, result[:-1])
                if not out:
                    continue
                relevancy = out['relevancy']
                print(f"Result {i+1}: {relevancy}, {result[1]} because: {out['comment']}")
                if self.search_engine == 'congress':
                    rel_docs[result[0]] = {'title':result[1], 'url':result[0], 'relevancy':relevancy, 'llm_comment':out['comment'], 'year': date, 'alternative_title':alt_title}
                else:
                    rel_docs[result[0]] = {'title':result[1], 'url':result[0], 'relevancy':relevancy, 'llm_comment':out['comment']}
                for key, value in out.items():
                    if key not in ['title', 'url', 'relevancy', 'comment']:
                        rel_docs[result[0]][key] = value
                
                if num_rel_chunks <= len(chunks):
                    for i in range(num_rel_chunks):
                        rel_docs[result[0]][f"Most Relevant Chunk {i+1}"] = chunks[i]
                else:
                    for i in range(len(chunks)):
                        if i < len(chunks):
                            rel_docs[result[0]][f"Most Relevant Chunk {i+1}"] = chunks[i]
                        else:
                            rel_docs[result[0]][f"Most Relevant Chunk {i+1}"] = "No more chunks available."
                print('--'*50)
            else:
                continue

        self.rel_docs = rel_docs
        self.df = pd.DataFrame.from_dict(rel_docs, orient='index').reset_index()
        self.df = self.df[self.df.columns[1:]]
        if (to_excel) and (len(self.df) > 0):
            now = datetime.now()
            dt_string = now.strftime("%m-%d-%Y")
            self.df.to_excel(f'./saved_searches/{self.query}_{self.max_results}_{dt_string}_results.xlsx')

        return self.df