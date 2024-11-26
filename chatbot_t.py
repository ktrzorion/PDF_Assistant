import os
import pandas as pd
import os
import requests
from tqdm import tqdm
from langchain_openai import ChatOpenAI
import asyncio
import json
from typing import List, Dict
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import CharacterTextSplitter
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_community.vectorstores import FAISS
from langchain.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from fuzzywuzzy import fuzz
import re
from datetime import datetime
from dotenv import load_dotenv
import time

# Load environment variables from .env file
load_dotenv()

class ProjectInfo:
    def __init__(self, name, drive_link, **kwargs):
        self.name = name
        self.drive_link = drive_link
        self.technologies = set()
        self.description = ""
        self.developers = set()
        self.file_source = kwargs.get('file_source', "")
        self.page_numbers = set()

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

llm = ChatOpenAI(model="gpt-4o-mini")

OUTPUT_FOLDER = 'pdfs'

if not os.path.exists(OUTPUT_FOLDER):
    os.makedirs(OUTPUT_FOLDER)

df = pd.read_excel('SalesData.xlsx')

for index, row in tqdm(df.iterrows(), total=len(df), desc='Downloading PDFs'):
    name = row['Name of Project']
    drive_url = row['Drive Link']
    pdf_filename = f'{name}.pdf'
    pdf_path = os.path.join(OUTPUT_FOLDER, pdf_filename)

    # Skip if the file is already downloaded
    if os.path.exists(pdf_path):
        # print(f"{pdf_filename} already exists. Skipping download.")
        continue

    # Check if drive_url is valid
    if not drive_url or not drive_url.startswith('https://drive.google.com/'):
        # print(f"Invalid drive_url: {drive_url} for {name}")
        continue

    # Extract file_id
    url_parts = drive_url.split('/')
    if len(url_parts) < 6:  # Expected format: https://drive.google.com/file/d/{file_id}/...
        # print(f"Unexpected drive_url format: {drive_url} for {name}")
        continue

    file_id = url_parts[-2]
    download_url = f'https://drive.google.com/uc?id={file_id}&export=download'

    try:
        response = requests.get(download_url, timeout=10)
        response.raise_for_status()  # Raise an exception for HTTP errors

        # Save the downloaded file
        with open(pdf_path, 'wb') as f:
            f.write(response.content)

        # print(f'Downloaded {pdf_filename}')
    except requests.RequestException as e:
        print(f'Error downloading {pdf_filename}: {e}')
        
    except Exception as e:
        print(f'Error saving {pdf_filename}: {e}')


class PersistentProjectProcessor:
    def __init__(self, folder_path: str, cache_dir: str = "project_cache"):
        self.folder_path = folder_path
        self.cache_dir = cache_dir
        self.embeddings = OpenAIEmbeddings(
            model="text-embedding-3-large",
            chunk_size=1000
        )
        self.llm = ChatOpenAI(
            model="gpt-4o",
            temperature=0
        )
        self.vector_store = None
        self.projects = {}
        self.technology_index = {}
        self.last_processed_time = None
        self.df = pd.read_excel('SalesData.xlsx')

        # Create cache directory if it doesn't exist
        os.makedirs(cache_dir, exist_ok=True)

        # File paths for cached data
        self.cache_paths = {
            'projects': os.path.join(cache_dir, 'projects.json'),
            'tech_index': os.path.join(cache_dir, 'tech_index.json'),
            'vector_store': os.path.join(cache_dir, 'vector_store.faiss'),
            'metadata': os.path.join(cache_dir, 'metadata.json')
        }

        # Enhanced technology patterns with more variations
        self.tech_patterns = {
            'frontend': r'\b(React|Angular|Vue|Next\.js|Svelte|jQuery|HTML5|CSS3|Bootstrap|Tailwind|Material-UI)\b',
            'backend': r'\b(Node\.js|Express|Django|Flask|Spring|Laravel|FastAPI|Rails|ASP\.NET|PHP|python)\b',
            'database': r'\b(MongoDB|PostgreSQL|MySQL|Redis|Cassandra|SQLite|Oracle|DynamoDB|Firebase|ElasticSearch)\b',
            'mobile': r'\b(iOS|Android|React Native|Flutter|Swift|Kotlin|Xamarin|Ionic|PWA|Mobile Web)\b',
            'devops': r'\b(Docker|Kubernetes|Jenkins|AWS|Azure|GCP|CircleCI|Travis|GitLab|Terraform)\b',
            'languages': r'\b(python|Python|JavaScript|TypeScript|Java|C\+\+|Go|Rust|Ruby|C#|Scala|RubyOnRails)\b'
        }

    def _find_drive_link(self, project_name: str) -> str:
        """
        Enhanced drive link finding with multiple matching strategies and logging
        """
        print(f"\nSearching drive link for project: {project_name}")
        
        try:
            # Strategy 1: Exact match (case-insensitive)
            exact_match = self.df[self.df['Name of Project'].str.lower() == project_name.lower()]
            if not exact_match.empty:
                drive_link = exact_match['Drive Link'].iloc[0]
                if pd.isna(drive_link) or str(drive_link).strip() == '':
                    print(f"Found exact match but drive link is empty for: {project_name}")
                else:
                    print(f"Found exact match with drive link for: {project_name}")
                    return drive_link

            # Strategy 2: Partial match (case-insensitive)
            partial_match = self.df[self.df['Name of Project'].str.lower().str.contains(project_name.lower(), na=False)]
            if not partial_match.empty:
                drive_link = partial_match['Drive Link'].iloc[0]
                if not pd.isna(drive_link) and str(drive_link).strip() != '':
                    print(f"Found partial match with drive link for: {project_name}")
                    return drive_link

            # Strategy 3: Fuzzy matching with detailed logging
            print(f"Trying fuzzy matching for: {project_name}")
            best_match_score = 0
            best_match_name = None
            best_match_link = None
            
            for idx, row in self.df.iterrows():
                if pd.isna(row['Name of Project']):
                    continue
                    
                excel_name = str(row['Name of Project'])
                score = fuzz.ratio(project_name.lower(), excel_name.lower())
                print(f"Comparing with '{excel_name}' - Score: {score}")
                
                if score > 80 and score > best_match_score:
                    if not pd.isna(row['Drive Link']) and str(row['Drive Link']).strip() != '':
                        best_match_score = score
                        best_match_name = excel_name
                        best_match_link = row['Drive Link']

            if best_match_link:
                print(f"Best fuzzy match found: '{best_match_name}' with score {best_match_score}")
                return best_match_link

            # Strategy 4: Try matching words individually
            project_words = set(project_name.lower().split())
            for idx, row in self.df.iterrows():
                if pd.isna(row['Name of Project']):
                    continue
                excel_name = str(row['Name of Project'])
                excel_words = set(excel_name.lower().split())
                word_overlap = len(project_words & excel_words)
                similarity = word_overlap / max(len(project_words), len(excel_words))
                if similarity > 0.5 and not pd.isna(row['Drive Link']) and row['Drive Link'].strip():
                    print(f"Found word-based match with similarity {similarity:.2f}: {excel_name}")
                    return row['Drive Link']
                
                # If more than 50% of words match
                matching_words = project_words.intersection(excel_words)
                if len(matching_words) / max(len(project_words), len(excel_words)) > 0.1:
                    if not pd.isna(row['Drive Link']) and str(row['Drive Link']).strip() != '':
                        print(f"Found word-based match: {excel_name}")
                        return row['Drive Link']

            print(f"No matching drive link found for: {project_name}")
            return "No drive link available"

        except Exception as e:
            print(f"Error finding drive link for {project_name}: {str(e)}")
            return "No drive link available"

    async def load_cached_data(self) -> bool:
        """Load processed data from cache if available and valid"""
        try:
            if not self._should_reprocess():
                # Load projects data
                with open(self.cache_paths['projects'], 'r') as f:
                    projects_data = json.load(f)
                    self.projects = {
                        name: ProjectInfo(
                            name=name,
                            technologies=set(data['technologies']),
                            description=data['description'],
                            developers=set(data['developers']),
                            drive_link=data['drive_link']
                        )
                        for name, data in projects_data.items()
                    }

                # Load technology index
                with open(self.cache_paths['tech_index'], 'r') as f:
                    tech_data = json.load(f)
                    self.technology_index = {
                        tech: set(projects)
                        for tech, projects in tech_data.items()
                    }

                # Load vector store
                if os.path.exists(self.cache_paths['vector_store']):
                    self.vector_store = FAISS.load_local(
                        self.cache_paths['vector_store'],
                        self.embeddings,
                        allow_dangerous_deserialization=True
                    )

                # print("Successfully loaded cached data!")
                return True

        except Exception as e:
            print(f"Error loading cached data: {str(e)}")

        return False

    async def save_cached_data(self):
        """Save processed data to cache"""
        try:
            # Save projects data
            projects_data = {
                name: {
                    'technologies': list(info.technologies),
                    'description': info.description,
                    'developers': list(info.developers),
                    'platforms': list(info.platforms),
                    'drive_link': info.drive_link
                }
                for name, info in self.projects.items()
            }
            with open(self.cache_paths['projects'], 'w') as f:
                json.dump(projects_data, f)

            # Save technology index
            tech_data = {
                tech: list(projects)
                for tech, projects in self.technology_index.items()
            }
            with open(self.cache_paths['tech_index'], 'w') as f:
                json.dump(tech_data, f)

            # Save vector store
            if self.vector_store:
                self.vector_store.save_local(self.cache_paths['vector_store'])

            # Save metadata
            metadata = {
                'pdf_times': self._get_pdf_modification_times(),
                'last_processed': datetime.now().isoformat()
            }
            with open(self.cache_paths['metadata'], 'w') as f:
                json.dump(metadata, f)

            # print("Successfully saved processed data to cache!")

        except Exception as e:
            print(f"Error saving cached data: {str(e)}")

    async def extract_project_info(self, text: str, filename: str, page_num: int) -> None:
        """Enhanced project information extraction"""
        # Expanded project patterns
        project_patterns = [
            r"Project\s*:\s*([^\n.]+)",
            r"Project Name\s*:\s*([^\n.]+)",
            r"Title\s*:\s*([^\n.]+)(?=\s*Project)",
            r"Application Name\s*:\s*([^\n.]+)",
            r"System Name\s*:\s*([^\n.]+)",
            r"Product Name\s*:\s*([^\n.]+)"
        ]

        for pattern in project_patterns:
            if match := re.search(pattern, text, re.IGNORECASE):
                project_name = match.group(1).strip()
                if project_name not in self.projects:
                    self.projects[project_name] = ProjectInfo(
                        name=project_name,
                        file_source=filename,
                        drive_link=row.get('Drive Link', '')
                    )
                project_info = self.projects[project_name]
                project_info.page_numbers.add(page_num)

                # Enhanced technology extraction
                for category, pattern in self.tech_patterns.items():
                    for tech in re.finditer(pattern, text, re.IGNORECASE):
                        tech_name = tech.group(0)
                        project_info.technologies.add(tech_name)
                        if tech_name not in self.technology_index:
                            self.technology_index[tech_name] = set()
                        self.technology_index[tech_name].add(project_name)

                # Enhanced description extraction with multiple patterns
                description_patterns = [
                    r"Description\s*:\s*([^\n.]+)",
                    r"Overview\s*:\s*([^\n.]+)",
                    r"Summary\s*:\s*([^\n.]+)",
                    r"About\s*:\s*([^\n.]+)"
                ]
                for desc_pattern in description_patterns:
                    if desc_match := re.search(desc_pattern, text, re.IGNORECASE):
                        project_info.description = desc_match.group(1).strip()
                        break

                # Enhanced developer/role extraction
                role_patterns = [
                    r"Developer[s]?\s*:\s*([^\n.]+)",
                    r"Team Member[s]?\s*:\s*([^\n.]+)",
                    r"Contributors?\s*:\s*([^\n.]+)",
                    r"Created by\s*:\s*([^\n.]+)"
                ]
                for role_pattern in role_patterns:
                    if role_match := re.findall(role_pattern, text, re.IGNORECASE):
                        project_info.developers.update(
                            dev.strip() for dev in role_match[0].split(',')
                        )

    def find_relevant_projects(self, query: str) -> List[Dict]:
        """Find all relevant projects with proper drive link handling"""
        scored_projects = []
        query_lower = query.lower()

        for project_name, project in self.projects.items():
            score = 0.0
            
            # Scoring logic remains the same...
            
            if score > 0:
                drive_link = self._find_drive_link(project_name)
                
                scored_projects.append({
                    "name": project_name,
                    "technologies": list(project.technologies),
                    "description": project.description,
                    "score": score,
                    "drive_link": drive_link if drive_link and drive_link.strip() else "No drive link available"
                })

        return sorted(scored_projects, key=lambda x: x['score'], reverse=True)

    async def query_projects(self, query: str) -> str:
        """Query with unlimited results"""
        if not self.vector_store:
            return "Please process PDFs first using process_pdfs()"

        normalized_query = query.strip().lower()
        
        timings = {}
        overall_start_time = time.time()
        step_start_time = time.time()

        relevant_projects = self.find_relevant_projects(normalized_query)
        timings["find_relevant_projects"] = time.time() - step_start_time

        step_start_time = time.time()
        docs = self.vector_store.similarity_search(normalized_query, k=40)
        context = "\n".join(doc.page_content for doc in docs)
        timings["vector_search"] = time.time() - step_start_time

        step_start_time = time.time()
        prompt = PromptTemplate(
            template=""" 
            Based on the following information, please provide a comprehensive answer about ALL relevant projects.

            Context from documents:
            {context}

            Relevant projects found (ranked by relevance):
            {projects}

            Question: {question}

            Important: Please include EVERY project that matches the query criteria, no matter how many there are.
            For each project, provide:
            - Project name
            - Technologies used
            - Brief description
            - Drive link (if available)

            Return the result in the following JSON format:
            [
                {{
                    "name": "Project Name",
                    "technologies": ["Technology1", "Technology2", ...],
                    "description": "Brief description of the project",
                    "drive_link": "URL of the project"
                }},
                ...
            ]

            If no matching projects are found, return the following JSON:
            {{
                "error": "No matching projects were found."
            }}

            Answer:
            """,
            input_variables=["context", "projects", "question"]
        )

        chain = (
            {
                "context": lambda _: context,
                "projects": lambda _: str(relevant_projects),
                "question": lambda _: normalized_query
            }
            | prompt
            | self.llm
            | StrOutputParser()
        )

        raw_res = await chain.ainvoke("")
        timings["pre_formatted"] = time.time() - step_start_time
        result = await self.format_and_fix_json(raw_res)
        timings["format_n_fix"] = time.time() - step_start_time

        # Add drive links from DataFrame if missing
        for project in result:
            print(project.get('drive_link'))
            # Handle empty, missing, or whitespace-only drive links
            if not project.get('drive_link') or str(project.get('drive_link')).strip() == '' or str(project.get('drive_link')).strip() == 'No drive link available' or str(project.get('drive_link')).strip() == 'URL of the project':
                project['drive_link'] = self._find_drive_link(project['name'])

        timings["response_postprocessing"] = time.time() - step_start_time
        timings["total_execution_time"] = time.time() - overall_start_time

        return result
    
    async def clean_and_parse_json(self, input_str):
        # Remove markdown code block indicators and extra content
        cleaned_input = re.sub(r'^```json?\s*', '', input_str, flags=re.MULTILINE)
        cleaned_input = re.sub(r'```\s*$', '', cleaned_input, flags=re.MULTILINE)

        try:
            # Parse the cleaned JSON string
            parsed_data = json.loads(cleaned_input)
            return parsed_data
        except json.JSONDecodeError as e:
            # print(f"Error decoding JSON: {e}")
            return None
    
    async def format_and_fix_json(self, json_string: str):
        if not json_string.strip():
            json_string = "[]"
        
        json_string = json_string.replace("Page numbers where the project is found", "")
        json_string = json_string.replace("Relevance score", "0")
        json_string = json_string.replace('""', '"No drive link available"')  # Replace empty strings
         
        try:
            cleaned_input = await self.clean_and_parse_json(json_string)
            
            if not isinstance(cleaned_input, list):
                raise ValueError("Expected a list of projects in the JSON")
            
            for item in cleaned_input:
                if not isinstance(item, dict):
                    raise ValueError(f"Each item in the JSON array must be an object: {item}")
                
                # Handle empty or missing drive links
                if not item.get('drive_link') or str(item.get('drive_link')).strip() == '':
                    item['drive_link'] = self._find_drive_link(item.get('name', ''))
                
                # Set other defaults
                item.setdefault("name", "Unknown Project")
                item.setdefault("technologies", ["Unknown Technology"])
                item.setdefault("description", "No description available.")
                
            return cleaned_input
        
        except json.JSONDecodeError as e:
            raise ValueError(f"Error decoding JSON: {e}")

    async def process_single_pdf(self, filename: str) -> List[str]:
        """Process a single PDF file"""
        file_path = os.path.join(self.folder_path, filename)
        chunks = []

        try:
            loader = PyPDFLoader(file_path)
            pages = await asyncio.get_event_loop().run_in_executor(
                None, loader.load
            )

            # Use CharacterTextSplitter for faster splitting
            splitter = CharacterTextSplitter(
                separator="\n",
                chunk_size=500,
                chunk_overlap=50,
                length_function=len
            )

            for i, page in enumerate(pages):
                # Clean text
                text = re.sub(r'\s+', ' ', page.page_content).strip()
                # Extract information
                await self.extract_project_info(text, filename, i + 1)
                # Split into chunks
                page_chunks = splitter.split_text(text)
                chunks.extend(page_chunks)

        except Exception as e:
            print(f"Error processing {filename}: {str(e)}")

        return chunks

    async def process_pdfs(self):
        """Process all PDFs in parallel"""
        pdf_files = [f for f in os.listdir(self.folder_path) if f.lower().endswith('.pdf')]

        # Process PDFs in parallel
        tasks = [self.process_single_pdf(pdf) for pdf in pdf_files]
        chunks_list = await asyncio.gather(*tasks)

        # Flatten chunks list
        all_chunks = [chunk for chunks in chunks_list for chunk in chunks]

        # Create FAISS index
        texts = [{"content": chunk, "source": "pdf"} for chunk in all_chunks]
        self.vector_store = FAISS.from_texts(
            [text["content"] for text in texts],
            self.embeddings,
            metadatas=texts
        )

    def _get_pdf_modification_times(self) -> Dict[str, float]:
        """Get modification times of all PDFs in the folder"""
        pdf_times = {}
        for file in os.listdir(self.folder_path):
            if file.lower().endswith('.pdf'):
                path = os.path.join(self.folder_path, file)
                pdf_times[file] = os.path.getmtime(path)
        return pdf_times
    
    def _should_reprocess(self) -> bool:
        """Check if PDFs need reprocessing by comparing modification times"""
        if not os.path.exists(self.cache_paths['metadata']):
            return True

        try:
            with open(self.cache_paths['metadata'], 'r') as f:
                metadata = json.load(f)
                cached_times = metadata.get('pdf_times', {})
                current_times = self._get_pdf_modification_times()

                return any(
                    current_times.get(file, 0) != cached_times.get(file, -1)
                    for file in set(current_times) | set(cached_times)
                )
        except:
            return True

    async def initialize(self):
      """Initialize the processor - either load cached data or process PDFs"""
      if not await self.load_cached_data():
          # print("Processing PDFs...")
          await self.process_pdfs()
          await self.save_cached_data()
          # print("Processing and caching complete!")
