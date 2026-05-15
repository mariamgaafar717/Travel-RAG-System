
import requests
from bs4 import BeautifulSoup
import json
import time
import re
import numpy as np
import faiss
from sentence_transformers import SentenceTransformer
import pickle

places = {

    # ───────────────── ENGLISH ─────────────────

    "Cairo": {
        "governorate": "Cairo",
        "language": "en",
        "places": {
            "Egyptian Museum": {
                "wiki_title": "Egyptian_Museum",
                "links": [
                    "https://en.wikipedia.org/wiki/Egyptian_Museum",
                    "https://www.egyptianmuseum.gov.eg/",
                    "https://www.britannica.com/topic/Egyptian-Museum",
                    "https://www.lonelyplanet.com/egypt/cairo/attractions/egyptian-museum",
                    "https://www.nationalgeographic.com/history/article/egypt-museum"
                ]
            },
            "Khan el-Khalili": {
                "wiki_title": "Khan_el-Khalili",
                "links": [
                    "https://en.wikipedia.org/wiki/Khan_el-Khalili",
                    "https://egypt.travel/en/attractions/khan-el-khalili",
                    "https://www.lonelyplanet.com/egypt/cairo/attractions/khan-al-khalili",
                    "https://theculturetrip.com/africa/egypt/articles/a-brief-history-of-khan-el-khalili",
                    "https://www.egypttoursportal.com/khan-el-khalili-bazaar/"
                ]
            },
            "Saladin Citadel": {
                "wiki_title": "Cairo_Citadel",
                "links": [
                    "https://en.wikipedia.org/wiki/Cairo_Citadel",
                    "https://egypt.travel/en/attractions/saladin-citadel",
                    "https://www.britannica.com/topic/Citadel-of-Cairo",
                    "https://www.lonelyplanet.com/egypt/cairo/attractions/citadel",
                    "https://www.egypttoursportal.com/cairo-citadel/"
                ]
            },
            "Al-Azhar Mosque": {
                "wiki_title": "Al-Azhar_Mosque",
                "links": [
                    "https://en.wikipedia.org/wiki/Al-Azhar_Mosque",
                    "https://egypt.travel/en/attractions/al-azhar-mosque",
                    "https://www.britannica.com/topic/al-Azhar-mosque",
                    "https://www.lonelyplanet.com/egypt/cairo/attractions/al-azhar-mosque",
                    "https://www.egypttoursportal.com/al-azhar-mosque/"
                ]
            },
            "Coptic Cairo": {
                "wiki_title": "Coptic_Cairo",
                "links": [
                    "https://en.wikipedia.org/wiki/Coptic_Cairo",
                    "https://egypt.travel/en/attractions/coptic-cairo",
                    "https://www.lonelyplanet.com/egypt/cairo/attractions/coptic-cairo",
                    "https://theculturetrip.com/africa/egypt/articles/a-guide-to-coptic-cairo",
                    "https://www.egypttoursportal.com/coptic-cairo/"
                ]
            },
        }
    },

    "Giza": {
        "governorate": "Giza",
        "language": "en",
        "places": {
            "Great Pyramid of Giza": {
                "wiki_title": "Great_Pyramid_of_Giza",
                "links": [
                    "https://en.wikipedia.org/wiki/Great_Pyramid_of_Giza",
                    "https://egypt.travel/en/attractions/the-great-pyramid-of-giza",
                    "https://www.britannica.com/topic/Great-Pyramid-of-Giza",
                    "https://www.nationalgeographic.com/history/article/great-pyramid-giza",
                    "https://whc.unesco.org/en/list/86/"
                ]
            },
            "Great Sphinx of Giza": {
                "wiki_title": "Great_Sphinx_of_Giza",
                "links": [
                    "https://en.wikipedia.org/wiki/Great_Sphinx_of_Giza",
                    "https://egypt.travel/en/attractions/the-sphinx",
                    "https://www.britannica.com/topic/Great-Sphinx",
                    "https://www.nationalgeographic.com/history/article/great-sphinx",
                    "https://www.worldhistory.org/Great_Sphinx_of_Giza/"
                ]
            },
            "Saqqara": {
                "wiki_title": "Saqqara",
                "links": [
                    "https://en.wikipedia.org/wiki/Saqqara",
                    "https://egypt.travel/en/attractions/saqqara",
                    "https://www.britannica.com/place/Saqqarah",
                    "https://www.nationalgeographic.com/history/article/saqqara",
                    "https://www.egypttoursportal.com/saqqara/"
                ]
            },
            "Dahshur": {
                "wiki_title": "Dahshur",
                "links": [
                    "https://en.wikipedia.org/wiki/Dahshur",
                    "https://egypt.travel/en/attractions/dahshur",
                    "https://www.britannica.com/place/Dahshur",
                    "https://www.lonelyplanet.com/egypt/around-cairo/attractions/dahshur",
                    "https://www.egypttoursportal.com/dahshur/"
                ]
            },
            "Memphis Egypt": {
                "wiki_title": "Memphis,_Egypt",
                "links": [
                    "https://en.wikipedia.org/wiki/Memphis,_Egypt",
                    "https://egypt.travel/en/attractions/memphis",
                    "https://www.britannica.com/place/Memphis-ancient-city-Egypt",
                    "https://www.worldhistory.org/Memphis/",
                    "https://www.egypttoursportal.com/memphis-egypt/"
                ]
            },
        }
    },

    "Alexandria": {
        "governorate": "Alexandria",
        "language": "en",
        "places": {
            "Bibliotheca Alexandrina": {
                "wiki_title": "Bibliotheca_Alexandrina",
                "links": [
                    "https://en.wikipedia.org/wiki/Bibliotheca_Alexandrina",
                    "https://www.bibalex.org/en/default",
                    "https://egypt.travel/en/attractions/bibliotheca-alexandrina",
                    "https://www.britannica.com/topic/Bibliotheca-Alexandrina",
                    "https://www.lonelyplanet.com/egypt/alexandria/attractions/bibliotheca-alexandrina"
                ]
            },
            "Qaitbay Citadel": {
                "wiki_title": "Citadel_of_Qaitbay",
                "links": [
                    "https://en.wikipedia.org/wiki/Citadel_of_Qaitbay",
                    "https://egypt.travel/en/attractions/qaitbay-citadel",
                    "https://www.britannica.com/topic/Fort-Qaitbey",
                    "https://www.lonelyplanet.com/egypt/alexandria/attractions/fort-qaitbey",
                    "https://www.egypttoursportal.com/qaitbay-citadel/"
                ]
            },
            "Pompeys Pillar": {
                "wiki_title": "Pompey%27s_Pillar",
                "links": [
                    "https://en.wikipedia.org/wiki/Pompey%27s_Pillar",
                    "https://egypt.travel/en/attractions/pompeys-pillar",
                    "https://www.britannica.com/topic/Pompeys-Pillar",
                    "https://www.lonelyplanet.com/egypt/alexandria/attractions/pompeys-pillar",
                    "https://www.egypttoursportal.com/pompeys-pillar/"
                ]
            },
            "Montaza Palace": {
                "wiki_title": "Montaza_Palace",
                "links": [
                    "https://en.wikipedia.org/wiki/Montaza_Palace",
                    "https://egypt.travel/en/attractions/montaza-palace",
                    "https://www.lonelyplanet.com/egypt/alexandria/attractions/montazah-palace-gardens",
                    "https://theculturetrip.com/africa/egypt/articles/a-history-of-montaza-palace",
                    "https://www.egypttoursportal.com/montaza-palace/"
                ]
            },
            "Catacombs of Kom el Shoqafa": {
                "wiki_title": "Catacombs_of_Kom_el_Shoqafa",
                "links": [
                    "https://en.wikipedia.org/wiki/Catacombs_of_Kom_el_Shoqafa",
                    "https://egypt.travel/en/attractions/catacombs-of-kom-el-shoqafa",
                    "https://www.britannica.com/place/Catacombs-of-Kom-es-Shuqafa",
                    "https://www.lonelyplanet.com/egypt/alexandria/attractions/catacombs-of-kom-ash-shuqqafa",
                    "https://www.worldhistory.org/Catacombs_of_Kom_el_Shoqafa/"
                ]
            },
        }
    },

    "Luxor": {
        "governorate": "Luxor",
        "language": "en",
        "places": {
            "Valley of the Kings": {
                "wiki_title": "Valley_of_the_Kings",
                "links": [
                    "https://en.wikipedia.org/wiki/Valley_of_the_Kings",
                    "https://www.britannica.com/place/Valley-of-the-Kings",
                    "https://www.nationalgeographic.com/history/article/valley-of-the-kings",
                    "https://whc.unesco.org/en/list/87/",
                    "https://www.worldhistory.org/Valley_of_the_Kings/"
                ]
            },
            "Karnak Temple": {
                "wiki_title": "Karnak",
                "links": [
                    "https://en.wikipedia.org/wiki/Karnak",
                    "https://www.britannica.com/topic/Karnak-temple-complex-Egypt",
                    "https://egypt.travel/en/attractions/karnak-temple",
                    "https://www.nationalgeographic.com/history/article/karnak",
                    "https://www.worldhistory.org/Karnak/"
                ]
            },
            "Luxor Temple": {
                "wiki_title": "Luxor_Temple",
                "links": [
                    "https://en.wikipedia.org/wiki/Luxor_Temple",
                    "https://egypt.travel/en/attractions/luxor-temple",
                    "https://www.britannica.com/topic/Luxor-Temple",
                    "https://www.lonelyplanet.com/egypt/luxor/attractions/luxor-temple",
                    "https://www.worldhistory.org/Luxor_Temple/"
                ]
            },
            "Valley of the Queens": {
                "wiki_title": "Valley_of_the_Queens",
                "links": [
                    "https://en.wikipedia.org/wiki/Valley_of_the_Queens",
                    "https://egypt.travel/en/attractions/valley-of-the-queens",
                    "https://www.britannica.com/place/Valley-of-the-Queens",
                    "https://whc.unesco.org/en/list/87/",
                    "https://www.egypttoursportal.com/valley-of-the-queens/"
                ]
            },
            "Mortuary Temple of Hatshepsut": {
                "wiki_title": "Mortuary_Temple_of_Hatshepsut",
                "links": [
                    "https://en.wikipedia.org/wiki/Mortuary_Temple_of_Hatshepsut",
                    "https://egypt.travel/en/attractions/hatshepsut-temple",
                    "https://www.britannica.com/topic/Deir-el-Bahri",
                    "https://www.nationalgeographic.com/history/article/hatshepsut",
                    "https://www.worldhistory.org/Mortuary_Temple_of_Hatshepsut/"
                ]
            },
        }
    },

    "Aswan": {
        "governorate": "Aswan",
        "language": "en",
        "places": {
            "Abu Simbel temples": {
                "wiki_title": "Abu_Simbel_temples",
                "links": [
                    "https://en.wikipedia.org/wiki/Abu_Simbel_temples",
                    "https://egypt.travel/en/attractions/abu-simbel",
                    "https://www.britannica.com/place/Abu-Simbel",
                    "https://whc.unesco.org/en/list/88/",
                    "https://www.nationalgeographic.com/history/article/abu-simbel"
                ]
            },
            "Philae Temple": {
                "wiki_title": "Philae",
                "links": [
                    "https://en.wikipedia.org/wiki/Philae",
                    "https://egypt.travel/en/attractions/philae-temple",
                    "https://www.britannica.com/place/Philae",
                    "https://whc.unesco.org/en/list/88/",
                    "https://www.worldhistory.org/Philae/"
                ]
            },
            "Aswan High Dam": {
                "wiki_title": "Aswan_High_Dam",
                "links": [
                    "https://en.wikipedia.org/wiki/Aswan_High_Dam",
                    "https://egypt.travel/en/attractions/aswan-high-dam",
                    "https://www.britannica.com/topic/Aswan-High-Dam",
                    "https://www.nationalgeographic.com/history/article/aswan-high-dam",
                    "https://www.egypttoursportal.com/aswan-high-dam/"
                ]
            },
            "Nubian Museum": {
                "wiki_title": "Nubian_Museum",
                "links": [
                    "https://en.wikipedia.org/wiki/Nubian_Museum",
                    "https://egypt.travel/en/attractions/nubian-museum",
                    "https://www.britannica.com/topic/Nubian-Museum",
                    "https://www.lonelyplanet.com/egypt/aswan/attractions/nubian-museum",
                    "https://www.egypttoursportal.com/nubian-museum/"
                ]
            },
            "Elephantine Island": {
                "wiki_title": "Elephantine",
                "links": [
                    "https://en.wikipedia.org/wiki/Elephantine",
                    "https://egypt.travel/en/attractions/elephantine-island",
                    "https://www.britannica.com/place/Elephantine",
                    "https://www.worldhistory.org/Elephantine/",
                    "https://www.egypttoursportal.com/elephantine-island/"
                ]
            },
        }
    },

    "Red Sea": {
        "governorate": "Red Sea",
        "language": "en",
        "places": {
            "Hurghada": {
                "wiki_title": "Hurghada",
                "links": [
                    "https://en.wikipedia.org/wiki/Hurghada",
                    "https://egypt.travel/en/attractions/hurghada",
                    "https://www.lonelyplanet.com/egypt/hurghada",
                    "https://www.divezone.net/diving/red-sea/hurghada",
                    "https://www.egypttoursportal.com/hurghada-excursions/"
                ]
            },
            "Marsa Alam": {
                "wiki_title": "Marsa_Alam",
                "links": [
                    "https://en.wikipedia.org/wiki/Marsa_Alam",
                    "https://egypt.travel/en/attractions/marsa-alam",
                    "https://www.lonelyplanet.com/egypt/marsa-alam",
                    "https://www.divezone.net/diving/red-sea/marsa-alam",
                    "https://www.egypttoursportal.com/marsa-alam-excursions/"
                ]
            },
            "El Gouna": {
                "wiki_title": "El_Gouna",
                "links": [
                    "https://en.wikipedia.org/wiki/El_Gouna",
                    "https://egypt.travel/en/attractions/el-gouna",
                    "https://www.lonelyplanet.com/egypt/el-gouna",
                    "https://www.elgouna.com/",
                    "https://theculturetrip.com/africa/egypt/articles/el-gouna-egypt"
                ]
            },
            "Safaga": {
                "wiki_title": "Safaga",
                "links": [
                    "https://en.wikipedia.org/wiki/Safaga",
                    "https://egypt.travel/en/attractions/safaga",
                    "https://www.divezone.net/diving/red-sea/safaga",
                    "https://www.lonelyplanet.com/egypt/safaga",
                    "https://www.egypttoursportal.com/safaga/"
                ]
            },
            "Ras Banas": {
                "wiki_title": "Ras_Banas",
                "links": [
                    "https://en.wikipedia.org/wiki/Ras_Banas",
                    "https://egypt.travel/en/attractions/ras-banas",
                    "https://www.divezone.net/diving/red-sea/ras-banas",
                    "https://www.lonelyplanet.com/egypt/ras-banas",
                    "https://www.egypttoursportal.com/ras-banas/"
                ]
            },
        }
    },

    "South Sinai": {
        "governorate": "South Sinai",
        "language": "en",
        "places": {
            "Sharm El Sheikh": {
                "wiki_title": "Sharm_el-Sheikh",
                "links": [
                    "https://en.wikipedia.org/wiki/Sharm_el-Sheikh",
                    "https://egypt.travel/en/attractions/sharm-el-sheikh",
                    "https://www.lonelyplanet.com/egypt/sinai/sharm-el-sheikh",
                    "https://www.divezone.net/diving/red-sea/sharm-el-sheikh",
                    "https://www.egypttoursportal.com/sharm-el-sheikh-excursions/"
                ]
            },
            "Dahab": {
                "wiki_title": "Dahab",
                "links": [
                    "https://en.wikipedia.org/wiki/Dahab",
                    "https://egypt.travel/en/attractions/dahab",
                    "https://www.lonelyplanet.com/egypt/sinai/dahab",
                    "https://www.divezone.net/diving/red-sea/dahab",
                    "https://www.egypttoursportal.com/dahab-excursions/"
                ]
            },
            "Ras Mohammed": {
                "wiki_title": "Ras_Mohammed_National_Park",
                "links": [
                    "https://en.wikipedia.org/wiki/Ras_Mohammed_National_Park",
                    "https://egypt.travel/en/attractions/ras-mohammed",
                    "https://www.lonelyplanet.com/egypt/sinai/ras-mohammed-national-park",
                    "https://www.divezone.net/diving/red-sea/ras-mohammed",
                    "https://www.egypttoursportal.com/ras-mohammed-national-park/"
                ]
            },
            "Mount Sinai": {
                "wiki_title": "Mount_Sinai",
                "links": [
                    "https://en.wikipedia.org/wiki/Mount_Sinai",
                    "https://egypt.travel/en/attractions/mount-sinai",
                    "https://www.britannica.com/place/Mount-Sinai",
                    "https://www.nationalgeographic.com/history/article/mount-sinai",
                    "https://www.lonelyplanet.com/egypt/sinai/mount-sinai"
                ]
            },
            "Taba Egypt": {
                "wiki_title": "Taba,_Egypt",
                "links": [
                    "https://en.wikipedia.org/wiki/Taba,_Egypt",
                    "https://egypt.travel/en/attractions/taba",
                    "https://www.lonelyplanet.com/egypt/sinai/taba",
                    "https://www.egypttoursportal.com/taba/",
                    "https://theculturetrip.com/africa/egypt/articles/taba-egypt"
                ]
            },
        }
    },

    "Matrouh": {
        "governorate": "Matrouh",
        "language": "en",
        "places": {
            "Siwa Oasis": {
                "wiki_title": "Siwa_Oasis",
                "links": [
                    "https://en.wikipedia.org/wiki/Siwa_Oasis",
                    "https://egypt.travel/en/attractions/siwa-oasis",
                    "https://www.lonelyplanet.com/egypt/western-desert/siwa-oasis",
                    "https://theculturetrip.com/africa/egypt/articles/a-guide-to-egypts-siwa-oasis/",
                    "https://www.egypttoursportal.com/siwa-oasis-tours/"
                ]
            },
            "Marsa Matrouh": {
                "wiki_title": "Marsa_Matruh",
                "links": [
                    "https://en.wikipedia.org/wiki/Marsa_Matruh",
                    "https://egypt.travel/en/attractions/marsa-matrouh",
                    "https://www.lonelyplanet.com/egypt/mediterranean-coast/marsa-matruh",
                    "https://theculturetrip.com/africa/egypt/articles/marsa-matrouh",
                    "https://www.egypttoursportal.com/marsa-matrouh/"
                ]
            },
            "El Alamein": {
                "wiki_title": "El_Alamein",
                "links": [
                    "https://en.wikipedia.org/wiki/El_Alamein",
                    "https://egypt.travel/en/attractions/el-alamein",
                    "https://www.britannica.com/place/El-Alamein",
                    "https://www.lonelyplanet.com/egypt/mediterranean-coast/el-alamein",
                    "https://www.egypttoursportal.com/el-alamein/"
                ]
            },
            "Agiba Beach": {
                "wiki_title": "Agiba_Beach",
                "links": [
                    "https://en.wikipedia.org/wiki/Agiba_Beach",
                    "https://egypt.travel/en/attractions/agiba-beach",
                    "https://www.lonelyplanet.com/egypt/mediterranean-coast/agiba-beach",
                    "https://theculturetrip.com/africa/egypt/articles/agiba-beach",
                    "https://www.egypttoursportal.com/agiba-beach/"
                ]
            },
            "Cleopatra Beach": {
                "wiki_title": "Cleopatra_Beach,_Matrouh",
                "links": [
                    "https://en.wikipedia.org/wiki/Cleopatra_Beach,_Matrouh",
                    "https://egypt.travel/en/attractions/cleopatra-beach",
                    "https://www.lonelyplanet.com/egypt/mediterranean-coast/cleopatra-beach",
                    "https://theculturetrip.com/africa/egypt/articles/cleopatra-beach",
                    "https://www.egypttoursportal.com/cleopatra-beach/"
                ]
            },
        }
    },

    "Fayoum": {
        "governorate": "Fayoum",
        "language": "en",
        "places": {
            "Lake Qarun": {
                "wiki_title": "Lake_Qarun",
                "links": [
                    "https://en.wikipedia.org/wiki/Lake_Qarun",
                    "https://egypt.travel/en/attractions/lake-qarun",
                    "https://www.britannica.com/place/Lake-Qarun",
                    "https://www.lonelyplanet.com/egypt/around-cairo/lake-qarun",
                    "https://www.egypttoursportal.com/lake-qarun/"
                ]
            },
            "Wadi El Rayan": {
                "wiki_title": "Wadi_El_Rayan",
                "links": [
                    "https://en.wikipedia.org/wiki/Wadi_El_Rayan",
                    "https://egypt.travel/en/attractions/wadi-el-rayan",
                    "https://www.lonelyplanet.com/egypt/around-cairo/wadi-el-rayan",
                    "https://theculturetrip.com/africa/egypt/articles/wadi-el-rayan",
                    "https://www.egypttoursportal.com/wadi-el-rayan/"
                ]
            },
            "Karanis": {
                "wiki_title": "Karanis",
                "links": [
                    "https://en.wikipedia.org/wiki/Karanis",
                    "https://egypt.travel/en/attractions/karanis",
                    "https://www.britannica.com/place/Karanis",
                    "https://www.worldhistory.org/Karanis/",
                    "https://www.egypttoursportal.com/karanis/"
                ]
            },
            "Hawara": {
                "wiki_title": "Hawara",
                "links": [
                    "https://en.wikipedia.org/wiki/Hawara",
                    "https://egypt.travel/en/attractions/hawara",
                    "https://www.britannica.com/place/Hawara",
                    "https://www.worldhistory.org/Hawara/",
                    "https://www.egypttoursportal.com/hawara/"
                ]
            },
            "Medinet el-Fayoum": {
                "wiki_title": "Faiyum",
                "links": [
                    "https://en.wikipedia.org/wiki/Faiyum",
                    "https://egypt.travel/en/attractions/fayoum",
                    "https://www.lonelyplanet.com/egypt/around-cairo/al-fayyum",
                    "https://theculturetrip.com/africa/egypt/articles/fayoum",
                    "https://www.egypttoursportal.com/fayoum/"
                ]
            },
        }
    },

    "Minya": {
        "governorate": "Minya",
        "language": "en",
        "places": {
            "Tell el-Amarna": {
                "wiki_title": "Amarna",
                "links": [
                    "https://en.wikipedia.org/wiki/Amarna",
                    "https://egypt.travel/en/attractions/amarna",
                    "https://www.britannica.com/place/Amarna-Egypt",
                    "https://www.worldhistory.org/Amarna/",
                    "https://www.egypttoursportal.com/amarna/"
                ]
            },
            "Beni Hassan": {
                "wiki_title": "Beni_Hassan",
                "links": [
                    "https://en.wikipedia.org/wiki/Beni_Hassan",
                    "https://egypt.travel/en/attractions/beni-hassan",
                    "https://www.britannica.com/place/Beni-Hasan",
                    "https://www.worldhistory.org/Beni_Hassan/",
                    "https://www.egypttoursportal.com/beni-hassan/"
                ]
            },
            "Hermopolis": {
                "wiki_title": "Hermopolis",
                "links": [
                    "https://en.wikipedia.org/wiki/Hermopolis",
                    "https://egypt.travel/en/attractions/hermopolis",
                    "https://www.britannica.com/place/Hermopolis",
                    "https://www.worldhistory.org/Hermopolis/",
                    "https://www.egypttoursportal.com/hermopolis/"
                ]
            },
            "Tuna el-Gebel": {
                "wiki_title": "Tuna_el-Gebel",
                "links": [
                    "https://en.wikipedia.org/wiki/Tuna_el-Gebel",
                    "https://egypt.travel/en/attractions/tuna-el-gebel",
                    "https://www.britannica.com/place/Tuna-el-Gebel",
                    "https://www.worldhistory.org/Tuna_el-Gebel/",
                    "https://www.egypttoursportal.com/tuna-el-gebel/"
                ]
            },
            "Speos Artemidos": {
                "wiki_title": "Speos_Artemidos",
                "links": [
                    "https://en.wikipedia.org/wiki/Speos_Artemidos",
                    "https://egypt.travel/en/attractions/speos-artemidos",
                    "https://www.britannica.com/place/Speos-Artemidos",
                    "https://www.worldhistory.org/Speos_Artemidos/",
                    "https://www.egypttoursportal.com/speos-artemidos/"
                ]
            },
        }
    },

    "Sohag": {
        "governorate": "Sohag",
        "language": "en",
        "places": {
            "Abydos Egypt": {
                "wiki_title": "Abydos,_Egypt",
                "links": [
                    "https://en.wikipedia.org/wiki/Abydos,_Egypt",
                    "https://egypt.travel/en/attractions/abydos",
                    "https://www.britannica.com/place/Abydos-Egypt",
                    "https://www.worldhistory.org/Abydos/",
                    "https://www.egypttoursportal.com/abydos/"
                ]
            },
            "Dendera Temple": {
                "wiki_title": "Dendera_Temple_complex",
                "links": [
                    "https://en.wikipedia.org/wiki/Dendera_Temple_complex",
                    "https://egypt.travel/en/attractions/dendera-temple",
                    "https://www.britannica.com/place/Dendera",
                    "https://www.worldhistory.org/Dendera/",
                    "https://www.egypttoursportal.com/dendera-temple/"
                ]
            },
            "White Monastery": {
                "wiki_title": "White_Monastery",
                "links": [
                    "https://en.wikipedia.org/wiki/White_Monastery",
                    "https://egypt.travel/en/attractions/white-monastery",
                    "https://www.britannica.com/place/White-Monastery",
                    "https://www.worldhistory.org/White_Monastery/",
                    "https://www.egypttoursportal.com/white-monastery/"
                ]
            },
            "Red Monastery": {
                "wiki_title": "Red_Monastery",
                "links": [
                    "https://en.wikipedia.org/wiki/Red_Monastery",
                    "https://egypt.travel/en/attractions/red-monastery",
                    "https://www.britannica.com/place/Red-Monastery",
                    "https://www.worldhistory.org/Red_Monastery/",
                    "https://www.egypttoursportal.com/red-monastery/"
                ]
            },
            "Osireion": {
                "wiki_title": "Osireion",
                "links": [
                    "https://en.wikipedia.org/wiki/Osireion",
                    "https://egypt.travel/en/attractions/osireion",
                    "https://www.britannica.com/place/Osireion",
                    "https://www.worldhistory.org/Osireion/",
                    "https://www.egypttoursportal.com/osireion/"
                ]
            },
        }
    },

    "Ismailia and Suez": {
        "governorate": "Ismailia and Suez",
        "language": "en",
        "places": {
            "Suez Canal": {
                "wiki_title": "Suez_Canal",
                "links": [
                    "https://en.wikipedia.org/wiki/Suez_Canal",
                    "https://egypt.travel/en/attractions/suez-canal",
                    "https://www.britannica.com/topic/Suez-Canal",
                    "https://www.nationalgeographic.com/history/article/suez-canal",
                    "https://www.worldhistory.org/Suez_Canal/"
                ]
            },
            "Ismailia": {
                "wiki_title": "Ismailia",
                "links": [
                    "https://en.wikipedia.org/wiki/Ismailia",
                    "https://egypt.travel/en/attractions/ismailia",
                    "https://www.lonelyplanet.com/egypt/suez-canal/ismailia",
                    "https://theculturetrip.com/africa/egypt/articles/ismailia",
                    "https://www.egypttoursportal.com/ismailia/"
                ]
            },
            "Port Said": {
                "wiki_title": "Port_Said",
                "links": [
                    "https://en.wikipedia.org/wiki/Port_Said",
                    "https://egypt.travel/en/attractions/port-said",
                    "https://www.britannica.com/place/Port-Said",
                    "https://www.lonelyplanet.com/egypt/suez-canal/port-said",
                    "https://www.egypttoursportal.com/port-said/"
                ]
            },
            "Timsah Lake": {
                "wiki_title": "Lake_Timsah",
                "links": [
                    "https://en.wikipedia.org/wiki/Lake_Timsah",
                    "https://egypt.travel/en/attractions/lake-timsah",
                    "https://www.britannica.com/place/Lake-Timsah",
                    "https://www.lonelyplanet.com/egypt/suez-canal/lake-timsah",
                    "https://www.egypttoursportal.com/lake-timsah/"
                ]
            },
            "Great Bitter Lake": {
                "wiki_title": "Great_Bitter_Lake",
                "links": [
                    "https://en.wikipedia.org/wiki/Great_Bitter_Lake",
                    "https://egypt.travel/en/attractions/great-bitter-lake",
                    "https://www.britannica.com/place/Great-Bitter-Lake",
                    "https://www.lonelyplanet.com/egypt/suez-canal/great-bitter-lake",
                    "https://www.egypttoursportal.com/great-bitter-lake/"
                ]
            },
        }
    },

    # ───────────────── ARABIC ─────────────────

    "القاهرة": {
        "governorate": "القاهرة",
        "language": "ar",
        "places": {
            "المتحف المصري": {
                "wiki_title": "المتحف_المصري",
                "links": [
                    "https://ar.wikipedia.org/wiki/المتحف_المصري",
                    "https://egypt.travel/ar/attractions/the-egyptian-museum",
                    "https://www.youm7.com/story/المتحف-المصري",
                    "https://www.masrawy.com/travel/destinations/المتحف-المصري",
                    "https://www.almasryalyoum.com/news/details/المتحف-المصري"
                ]
            },
            "خان الخليلي": {
                "wiki_title": "خان_الخليلي",
                "links": [
                    "https://ar.wikipedia.org/wiki/خان_الخليلي",
                    "https://egypt.travel/ar/attractions/khan-el-khalili",
                    "https://www.youm7.com/story/خان-الخليلي",
                    "https://www.masrawy.com/travel/destinations/خان-الخليلي",
                    "https://www.almasryalyoum.com/news/details/خان-الخليلي"
                ]
            },
            "قلعة صلاح الدين": {
                "wiki_title": "قلعة_صلاح_الدين_الأيوبي",
                "links": [
                    "https://ar.wikipedia.org/wiki/قلعة_صلاح_الدين_الأيوبي",
                    "https://egypt.travel/ar/attractions/saladin-citadel",
                    "https://www.youm7.com/story/قلعة-صلاح-الدين",
                    "https://www.masrawy.com/travel/destinations/قلعة-صلاح-الدين",
                    "https://www.almasryalyoum.com/news/details/قلعة-القاهرة"
                ]
            },
            "الجامع الأزهر": {
                "wiki_title": "الجامع_الأزهر",
                "links": [
                    "https://ar.wikipedia.org/wiki/الجامع_الأزهر",
                    "https://egypt.travel/ar/attractions/al-azhar-mosque",
                    "https://www.youm7.com/story/الجامع-الأزهر",
                    "https://www.masrawy.com/travel/destinations/الأزهر",
                    "https://www.almasryalyoum.com/news/details/الجامع-الأزهر"
                ]
            },
            "مصر القديمة": {
                "wiki_title": "مصر_القديمة",
                "links": [
                    "https://ar.wikipedia.org/wiki/مصر_القديمة",
                    "https://egypt.travel/ar/attractions/coptic-cairo",
                    "https://www.youm7.com/story/مصر-القديمة",
                    "https://www.masrawy.com/travel/destinations/مصر-القديمة",
                    "https://www.almasryalyoum.com/news/details/القاهرة-القبطية"
                ]
            },
        }
    },

    "الجيزة": {
        "governorate": "الجيزة",
        "language": "ar",
        "places": {
            "هرم خوفو": {
                "wiki_title": "هرم_خوفو",
                "links": [
                    "https://ar.wikipedia.org/wiki/هرم_خوفو",
                    "https://egypt.travel/ar/attractions/pyramid-of-khufu",
                    "https://www.youm7.com/story/هرم-خوفو",
                    "https://www.masrawy.com/travel/destinations/أهرامات-الجيزة",
                    "https://www.almasryalyoum.com/news/details/الهرم-الأكبر"
                ]
            },
            "أبو الهول": {
                "wiki_title": "أبو_الهول",
                "links": [
                    "https://ar.wikipedia.org/wiki/أبو_الهول",
                    "https://egypt.travel/ar/attractions/the-sphinx",
                    "https://www.youm7.com/story/أبو-الهول",
                    "https://www.masrawy.com/travel/destinations/أبو-الهول",
                    "https://www.almasryalyoum.com/news/details/تمثال-أبو-الهول"
                ]
            },
            "سقارة": {
                "wiki_title": "سقارة",
                "links": [
                    "https://ar.wikipedia.org/wiki/سقارة",
                    "https://egypt.travel/ar/attractions/saqqara",
                    "https://www.youm7.com/story/سقارة",
                    "https://www.masrawy.com/travel/destinations/سقارة",
                    "https://www.almasryalyoum.com/news/details/منطقة-سقارة"
                ]
            },
            "دهشور": {
                "wiki_title": "دهشور",
                "links": [
                    "https://ar.wikipedia.org/wiki/دهشور",
                    "https://egypt.travel/ar/attractions/dahshur",
                    "https://www.youm7.com/story/دهشور",
                    "https://www.masrawy.com/travel/destinations/دهشور",
                    "https://www.almasryalyoum.com/news/details/أهرامات-دهشور"
                ]
            },
            "منف": {
                "wiki_title": "مَنف",
                "links": [
                    "https://ar.wikipedia.org/wiki/مَنف",
                    "https://egypt.travel/ar/attractions/memphis",
                    "https://www.youm7.com/story/منف-مصر",
                    "https://www.masrawy.com/travel/destinations/منف",
                    "https://www.almasryalyoum.com/news/details/مدينة-منف"
                ]
            },
        }
    },

    "الأقصر": {
        "governorate": "الأقصر",
        "language": "ar",
        "places": {
            "وادي الملوك": {
                "wiki_title": "وادي_الملوك",
                "links": [
                    "https://ar.wikipedia.org/wiki/وادي_الملوك",
                    "https://egypt.travel/ar/attractions/valley-of-the-kings",
                    "https://www.youm7.com/story/وادي-الملوك",
                    "https://www.masrawy.com/travel/destinations/وادي-الملوك",
                    "https://www.almasryalyoum.com/news/details/وادي-الملوك-الأقصر"
                ]
            },
            "معبد الكرنك": {
                "wiki_title": "معبد_الكرنك",
                "links": [
                    "https://ar.wikipedia.org/wiki/معبد_الكرنك",
                    "https://egypt.travel/ar/attractions/karnak-temple",
                    "https://www.youm7.com/story/معبد-الكرنك",
                    "https://www.masrawy.com/travel/destinations/الكرنك",
                    "https://www.almasryalyoum.com/news/details/معبد-الكرنك"
                ]
            },
            "معبد الأقصر": {
                "wiki_title": "معبد_الأقصر",
                "links": [
                    "https://ar.wikipedia.org/wiki/معبد_الأقصر",
                    "https://egypt.travel/ar/attractions/luxor-temple",
                    "https://www.youm7.com/story/معبد-الأقصر",
                    "https://www.masrawy.com/travel/destinations/معبد-الأقصر",
                    "https://www.almasryalyoum.com/news/details/معبد-الأقصر"
                ]
            },
            "وادي الملكات": {
                "wiki_title": "وادي_الملكات",
                "links": [
                    "https://ar.wikipedia.org/wiki/وادي_الملكات",
                    "https://egypt.travel/ar/attractions/valley-of-the-queens",
                    "https://www.youm7.com/story/وادي-الملكات",
                    "https://www.masrawy.com/travel/destinations/وادي-الملكات",
                    "https://www.almasryalyoum.com/news/details/وادي-الملكات"
                ]
            },
            "معبد حتشبسوت": {
                "wiki_title": "معبد_حتشبسوت",
                "links": [
                    "https://ar.wikipedia.org/wiki/معبد_حتشبسوت",
                    "https://egypt.travel/ar/attractions/hatshepsut-temple",
                    "https://www.youm7.com/story/معبد-حتشبسوت",
                    "https://www.masrawy.com/travel/destinations/حتشبسوت",
                    "https://www.almasryalyoum.com/news/details/معبد-الدير-البحري"
                ]
            },
        }
    },

    "أسوان": {
        "governorate": "أسوان",
        "language": "ar",
        "places": {
            "معبد أبو سمبل": {
                "wiki_title": "معبد_أبو_سمبل",
                "links": [
                    "https://ar.wikipedia.org/wiki/معبد_أبو_سمبل",
                    "https://egypt.travel/ar/attractions/abu-simbel",
                    "https://www.youm7.com/story/أبو-سمبل",
                    "https://www.masrawy.com/travel/destinations/أبو-سمبل",
                    "https://www.almasryalyoum.com/news/details/معبد-أبو-سمبل"
                ]
            },
            "معبد فيلة": {
                "wiki_title": "معبد_فيلة",
                "links": [
                    "https://ar.wikipedia.org/wiki/معبد_فيلة",
                    "https://egypt.travel/ar/attractions/philae-temple",
                    "https://www.youm7.com/story/معبد-فيلة",
                    "https://www.masrawy.com/travel/destinations/فيلة",
                    "https://www.almasryalyoum.com/news/details/معبد-فيلة-أسوان"
                ]
            },
            "السد العالي": {
                "wiki_title": "السد_العالي",
                "links": [
                    "https://ar.wikipedia.org/wiki/السد_العالي",
                    "https://egypt.travel/ar/attractions/aswan-high-dam",
                    "https://www.youm7.com/story/السد-العالي",
                    "https://www.masrawy.com/travel/destinations/السد-العالي",
                    "https://www.almasryalyoum.com/news/details/السد-العالي-أسوان"
                ]
            },
            "المتحف النوبي": {
                "wiki_title": "متحف_النوبة",
                "links": [
                    "https://ar.wikipedia.org/wiki/متحف_النوبة",
                    "https://egypt.travel/ar/attractions/nubian-museum",
                    "https://www.youm7.com/story/المتحف-النوبي",
                    "https://www.masrawy.com/travel/destinations/المتحف-النوبي",
                    "https://www.almasryalyoum.com/news/details/المتحف-النوبي-أسوان"
                ]
            },
            "جزيرة الفنتين": {
                "wiki_title": "جزيرة_الفنتين",
                "links": [
                    "https://ar.wikipedia.org/wiki/جزيرة_الفنتين",
                    "https://egypt.travel/ar/attractions/elephantine-island",
                    "https://www.youm7.com/story/جزيرة-الفنتين",
                    "https://www.masrawy.com/travel/destinations/الفنتين",
                    "https://www.almasryalyoum.com/news/details/جزيرة-الفنتين"
                ]
            },
        }
    },

    "البحر الأحمر": {
        "governorate": "البحر الأحمر",
        "language": "ar",
        "places": {
            "الغردقة": {
                "wiki_title": "الغردقة",
                "links": [
                    "https://ar.wikipedia.org/wiki/الغردقة",
                    "https://egypt.travel/ar/attractions/hurghada",
                    "https://www.youm7.com/story/الغردقة",
                    "https://www.masrawy.com/travel/destinations/الغردقة",
                    "https://www.almasryalyoum.com/news/details/الغردقة-سياحة"
                ]
            },
            "مرسى علم": {
                "wiki_title": "مرسى_علم",
                "links": [
                    "https://ar.wikipedia.org/wiki/مرسى_علم",
                    "https://egypt.travel/ar/attractions/marsa-alam",
                    "https://www.youm7.com/story/مرسى-علم",
                    "https://www.masrawy.com/travel/destinations/مرسى-علم",
                    "https://www.almasryalyoum.com/news/details/مرسى-علم"
                ]
            },
            "الجونة": {
                "wiki_title": "الجونة",
                "links": [
                    "https://ar.wikipedia.org/wiki/الجونة",
                    "https://egypt.travel/ar/attractions/el-gouna",
                    "https://www.youm7.com/story/الجونة",
                    "https://www.masrawy.com/travel/destinations/الجونة",
                    "https://www.almasryalyoum.com/news/details/مدينة-الجونة"
                ]
            },
            "سفاجا": {
                "wiki_title": "سفاجا",
                "links": [
                    "https://ar.wikipedia.org/wiki/سفاجا",
                    "https://egypt.travel/ar/attractions/safaga",
                    "https://www.youm7.com/story/سفاجا",
                    "https://www.masrawy.com/travel/destinations/سفاجا",
                    "https://www.almasryalyoum.com/news/details/سفاجا-البحر-الأحمر"
                ]
            },
            "رأس بناس": {
                "wiki_title": "رأس_بناس",
                "links": [
                    "https://ar.wikipedia.org/wiki/رأس_بناس",
                    "https://egypt.travel/ar/attractions/ras-banas",
                    "https://www.youm7.com/story/رأس-بناس",
                    "https://www.masrawy.com/travel/destinations/رأس-بناس",
                    "https://www.almasryalyoum.com/news/details/رأس-بناس"
                ]
            },
        }
    },
}

print(f"Total governorates : {len(places)}")
print(f"Total places       : {sum(len(v['places']) for v in places.values())}")

def clean_text(text):
    text = re.sub(r"\[\d+\]", "", text)
    text = re.sub(r"\s+", " ", text)
    return text.strip()

def normalize_arabic(text):
    text = re.sub(r'[\u0617-\u061A\u064B-\u065F]', '', text)  # remove tashkeel
    text = re.sub(r'[أإآٱ]', 'ا', text)                        # unify alef
    text = re.sub(r'ة', 'ه', text)                             # ta marbuta
    text = re.sub(r'ى', 'ي', text)                             # alef maqsura
    text = re.sub(r'ـ', '', text)                              # remove tatweel
    text = re.sub(r"\[\d+\]", "", text)                        # citations
    text = re.sub(r"\s+", " ", text)
    return text.strip()

# Test
sample = "القَاهِرَةُ عاصِمَةُ مِصرَ"
print("Original:  ", sample)
print("Normalized:", normalize_arabic(sample))

def get_text(wiki_title, language="en"):
    if language == "ar":
        url = f"https://ar.wikipedia.org/wiki/{wiki_title}"
    else:
        url = f"https://en.wikipedia.org/wiki/{wiki_title}"

    headers = {"User-Agent": "Mozilla/5.0"}

    try:
        response = requests.get(url, headers=headers, timeout=10)

        if response.status_code != 200:
            print(f"    ✗ HTTP {response.status_code}: {wiki_title}")
            return None

        soup = BeautifulSoup(response.text, "html.parser")

        for tag in soup.find_all(["table", "sup", "style", "script"]):
            tag.decompose()

        content = soup.find_all(["p", "li"])
        text = " ".join([el.get_text() for el in content])

        if language == "ar":
            return normalize_arabic(text)
        else:
            return clean_text(text)

    except Exception as e:
        print(f"    ✗ Error {wiki_title}: {e}")
        return None

dataset = []

for governorate_name, gov_data in places.items():
    language = gov_data["language"]
    print(f"\n{'='*50}")
    print(f"  Governorate: {governorate_name} [{language.upper()}]")
    print(f"{'='*50}")

    for place_name, place_info in gov_data["places"].items():
        wiki_title = place_info["wiki_title"]
        links      = place_info.get("links", [])

        print(f"  Scraping: {place_name} ...")
        text = get_text(wiki_title, language=language)

        if text and len(text.split()) > 50:
            dataset.append({
                "place":       place_name,
                "city":        place_name,
                "governorate": governorate_name,
                "language":    language,
                "type":        "tourism",
                "text":        text,
                "links":       links,
                "source":      links[0] if links else ""
            })
            print(f" Done — {len(text.split())} words")
        else:
            print(f" Skipped (no content)")

        time.sleep(1)

with open("egypt_places.json", "w", encoding="utf-8") as f:
    json.dump(dataset, f, ensure_ascii=False, indent=4)

print(f"\nDataset saved: {len(dataset)} places total")

def chunk_text(text, chunk_size=100, overlap=20):
    words = text.split()
    chunks = []
    for i in range(0, len(words), chunk_size - overlap):
        chunk = " ".join(words[i:i + chunk_size])
        if len(chunk.split()) >= 20:      # skip tiny leftover chunks
            chunks.append(chunk)
    return chunks

chunked_data = []

for item in dataset:
    chunks = chunk_text(item["text"])
    for i, chunk in enumerate(chunks):
        chunked_data.append({
            "place":       item["place"],
            "city":        item["city"],
            "governorate": item["governorate"],
            "language":    item["language"],
            "type":        item["type"],
            "chunk_id":    i,
            "text":        chunk,
            "links":       item.get("links", []),
            "source":      item.get("source", "")
        })

with open("chunked_egypt_places.json", "w", encoding="utf-8") as f:
    json.dump(chunked_data, f, ensure_ascii=False, indent=4)

print(f"Chunking done: {len(chunked_data)} chunks total")

# Multilingual model — handles English + Arabic + cross-lingual search
model = SentenceTransformer("paraphrase-multilingual-MiniLM-L12-v2")

texts = [item["text"] for item in chunked_data]

print(f"Encoding {len(texts)} chunks ...")
embeddings = model.encode(texts, show_progress_bar=True, batch_size=64)

for i, item in enumerate(chunked_data):
    item["embedding"] = embeddings[i].tolist()

with open("embedded_egypt_places.json", "w", encoding="utf-8") as f:
    json.dump(chunked_data, f, ensure_ascii=False, indent=4)

print(f"Embeddings saved — shape: {embeddings.shape}")

embeddings_array = np.array(embeddings).astype("float32")

dimension = embeddings_array.shape[1]
index = faiss.IndexFlatL2(dimension)
index.add(embeddings_array)

print(f"FAISS index created")
print(f"  Vectors   : {index.ntotal}")
print(f"  Dimensions: {dimension}")

faiss.write_index(index, "faiss_index.index")

with open("faiss_metadata.pkl", "wb") as f:
    pickle.dump(chunked_data, f)

print("FAISS + Metadata saved")

def search(query, top_k=5):

    arabic_chars = re.search(r'[\u0600-\u06FF]', query)

    if arabic_chars:
        query = normalize_arabic(query)

    query_vec = model.encode([query]).astype("float32")

    faiss.normalize_L2(query_vec)

    distances, indices = index.search(query_vec, top_k)

    for rank, idx in enumerate(indices[0]):

        item = chunked_data[idx]
        dist = distances[0][rank]

        flag = "AR" if item["language"] == "ar" else "EN"

        print(f"\nRank {rank+1} {flag} | Score: {dist:.4f}")
        print(f"Place      : {item['place']}")
        print(f"Governorate: {item['governorate']}")
        print(f"Language   : {item['language']}")
        print(f"Source     : {item['source']}")
        print(f"Text       : {item['text'][:200]}...")